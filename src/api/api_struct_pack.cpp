#include "api_internal.h"

#include <cmath>
#include <cstring>
#include <string>

namespace brokernel::api {

namespace {

bool readNumberProp(Value obj, const char* key, double& out) {
    Value v = ev::getProperty(obj, key);
    if (!ev::isNumber(v) && !ev::isBool(v)) return false;
    out = ev::toDouble(v);
    return true;
}

bool writeScalarBytes(ScalarKind sk, Value v, uint8_t* dst, const std::string& ctx, std::string& err) {
    if (sk == ScalarKind::Bool) {
        if (!ev::isBool(v) && !ev::isNumber(v)) {
            err = ctx + ": expected boolean or number";
            return false;
        }
        const int32_t bv = ev::toBool(v) ? 1 : 0;
        std::memcpy(dst, &bv, 4);
        return true;
    }
    if (sk == ScalarKind::I64 || sk == ScalarKind::U64) {
        if (!ev::isNumber(v) && !ev::isBigInt(v) && !ev::isBool(v)) {
            err = ctx + ": expected integer or bigint";
            return false;
        }
        if (sk == ScalarKind::I64) {
            const int64_t x = ev::toInt64(v);
            std::memcpy(dst, &x, 8);
        } else {
            const uint64_t x = ev::toUint64(v);
            std::memcpy(dst, &x, 8);
        }
        return true;
    }
    if (!ev::isNumber(v) && !ev::isBool(v)) {
        err = ctx + ": expected number";
        return false;
    }
    const double d = ev::toDouble(v);
    switch (sk) {
        case ScalarKind::F32: {
            const float f = static_cast<float>(d);
            std::memcpy(dst, &f, 4);
            return true;
        }
        case ScalarKind::F64: {
            std::memcpy(dst, &d, 8);
            return true;
        }
        case ScalarKind::DF32: {
            const float hi = static_cast<float>(d);
            const float lo = std::isfinite(d) ? static_cast<float>(d - static_cast<double>(hi)) : 0.0f;
            std::memcpy(dst, &hi, 4);
            std::memcpy(dst + 4, &lo, 4);
            return true;
        }
        case ScalarKind::I32: {
            const int32_t x = static_cast<int32_t>(static_cast<int64_t>(d));
            std::memcpy(dst, &x, 4);
            return true;
        }
        case ScalarKind::U32: {
            const uint32_t x = static_cast<uint32_t>(static_cast<uint64_t>(d));
            std::memcpy(dst, &x, 4);
            return true;
        }
        default:
            err = ctx + ": unsupported scalar field type";
            return false;
    }
}

bool packVectorValue(const Type& type, Value v, uint8_t* dst, size_t dstBytes,
                     const std::string& ctx, std::string& err) {
    const uint32_t lanes = type.dimension;
    const uint32_t sb = type.scalar_size_bytes();
    const size_t need = static_cast<size_t>(lanes) * sb;
    if (dstBytes < need) {
        err = ctx + ": destination buffer too small for vector";
        return false;
    }
    if (ev::isTypedArray(v)) {
        const auto info = ev::typedArrayInfo(v);
        if (info.byteLength >= need && !info.data) {
            err = ctx + ": detached TypedArray";
            return false;
        }
        const uint32_t elemBytes = sb;
        const size_t count = elemBytes ? (info.byteLength / elemBytes) : 0;
        if (info.byteLength == need) {
            std::memcpy(dst, info.data, need);
            return true;
        }
        (void)count;
        err = ctx + ": TypedArray byte length " + std::to_string(info.byteLength) +
              " does not match expected " + std::to_string(need) + " bytes";
        return false;
    }
    if (!ev::isObject(v)) {
        err = ctx + ": vector expects an array, TypedArray, or {x,y,z,w} object";
        return false;
    }
    Value lenV = ev::getProperty(v, "length");
    if (ev::isNumber(lenV)) {
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
        if (n != lanes) {
            err = ctx + ": expected " + std::to_string(lanes) + " vector lane(s), got " + std::to_string(n);
            return false;
        }
        for (uint32_t i = 0; i < lanes; ++i) {
            ev::Persistent elem(ev::getElement(v, i));
            if (!writeScalarBytes(type.scalar, elem.get(), dst + i * sb,
                                  ctx + "[" + std::to_string(i) + "]", err)) {
                return false;
            }
        }
        return true;
    }
    static constexpr const char* kKeys[4] = {"x", "y", "z", "w"};
    double vals[4]{};
    for (uint32_t i = 0; i < lanes; ++i) {
        if (!readNumberProp(v, kKeys[i], vals[i])) {
            err = ctx + ": vector object missing numeric property '" + std::string(kKeys[i]) + "'";
            return false;
        }
    }
    for (uint32_t i = 0; i < lanes; ++i) {
        ev::Persistent num(ev::fromDouble(vals[i]));
        if (!writeScalarBytes(type.scalar, num.get(), dst + i * sb,
                              ctx + "." + std::string(kKeys[i]), err)) {
            return false;
        }
    }
    return true;
}

} // namespace

bool packJsStructValue(const TypeRegistry& types, const Type& type, Value v,
                       uint8_t* dst, size_t dstBytes, const std::string& ctx, std::string& err) {
    if (type.is_newtype()) {
        return packJsStructValue(types, types.unwrap_newtype(type), v, dst, dstBytes, ctx, err);
    }
    if (type.is_scalar()) {
        const uint32_t sb = type.scalar_size_bytes();
        if (dstBytes < sb) {
            err = ctx + ": destination buffer too small for scalar";
            return false;
        }
        return writeScalarBytes(type.scalar, v, dst, ctx, err);
    }
    if (type.is_vector()) {
        return packVectorValue(type, v, dst, dstBytes, ctx, err);
    }
    if (type.is_array()) {
        const Type elem = type.element_type();
        const uint32_t stride = types.byte_size(elem);
        const size_t total = static_cast<size_t>(stride) * type.dimension;
        if (dstBytes < total) {
            err = ctx + ": destination buffer too small for array";
            return false;
        }
        if (ev::isTypedArray(v)) {
            const auto info = ev::typedArrayInfo(v);
            if (!info.data || info.byteLength != total) {
                err = ctx + ": TypedArray byte length mismatch for array (expected " +
                      std::to_string(total) + " bytes)";
                return false;
            }
            std::memcpy(dst, info.data, total);
            return true;
        }
        if (!ev::isObject(v)) {
            err = ctx + ": fixed array expects a JS array or TypedArray";
            return false;
        }
        const Value lenV = ev::getProperty(v, "length");
        if (!ev::isNumber(lenV)) {
            err = ctx + ": fixed array object missing numeric length";
            return false;
        }
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
        if (n != type.dimension) {
            err = ctx + ": expected array of length " + std::to_string(type.dimension) +
                  ", got " + std::to_string(n);
            return false;
        }
        for (uint32_t i = 0; i < type.dimension; ++i) {
            ev::Persistent el(ev::getElement(v, i));
            if (!packJsStructValue(types, elem, el.get(), dst + i * stride, stride,
                                   ctx + "[" + std::to_string(i) + "]", err)) {
                return false;
            }
        }
        return true;
    }
    if (type.is_struct()) {
        const StructDef* sdef = types.find_struct(type.struct_id);
        if (!sdef) {
            err = ctx + ": unknown struct definition";
            return false;
        }
        if (dstBytes < sdef->byte_size) {
            err = ctx + ": destination buffer too small for struct '" + sdef->name + "'";
            return false;
        }
        if (ev::isTypedArray(v)) {
            const auto info = ev::typedArrayInfo(v);
            if (!info.data || info.byteLength != sdef->byte_size) {
                err = ctx + ": TypedArray byte length mismatch for struct '" + sdef->name + "'";
                return false;
            }
            std::memcpy(dst, info.data, sdef->byte_size);
            return true;
        }
        if (!ev::isObject(v)) {
            err = ctx + ": struct '" + sdef->name + "' expects a plain JS object or TypedArray";
            return false;
        }
        for (const StructField& f : sdef->fields) {
            const uint32_t fBytes = types.byte_size(f.type);
            if (f.byte_offset + fBytes > dstBytes) {
                err = ctx + ": struct field '" + f.name + "' out of bounds";
                return false;
            }
            ev::Persistent fv(ev::getProperty(v, f.name));
            if (ev::isUndefined(fv.get()) || ev::isNull(fv.get())) {
                err = ctx + ": missing or null field '" + f.name + "' on struct '" + sdef->name + "'";
                return false;
            }
            if (!packJsStructValue(types, f.type, fv.get(), dst + f.byte_offset, fBytes,
                                   ctx + "." + f.name, err)) {
                return false;
            }
        }
        return true;
    }
    err = ctx + ": unsupported type '" + type.to_string() + "' in struct pack";
    return false;
}

} // namespace brokernel::api
