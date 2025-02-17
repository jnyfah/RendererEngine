#pragma once
#include <Helpers/IntrusivePtr.h>
#include <ZEngineDef.h>
#include <atomic>
#include <future>
#include <mutex>
#include <string>

#define REPORT_PROGRESS(ctx, value)          \
    {                                        \
        if (m_progress_callback)             \
        {                                    \
            m_progress_callback(ctx, value); \
        }                                    \
    }

namespace Tetragrama::Serializers
{
    template <typename TSerializerData>
    struct Serializer : public ZEngine::Helpers::RefCounted
    {
    protected:
        typedef void (*on_serializer_complete_fn)(void* const);
        typedef void (*on_serializer_deserialize_complete_fn)(void* const, TSerializerData&& data);
        typedef void (*on_serializer_progress_fn)(void* const, float progress);
        typedef void (*on_serializer_error_fn)(void* const, std::string_view error_message);
        typedef void (*on_serializer_log_fn)(void* const, std::string_view log_message);

        on_serializer_deserialize_complete_fn m_deserialize_complete_callback{nullptr};
        on_serializer_complete_fn             m_complete_callback{nullptr};
        on_serializer_progress_fn             m_progress_callback{nullptr};
        on_serializer_error_fn                m_error_callback{nullptr};
        on_serializer_log_fn                  m_log_callback{nullptr};

        std::recursive_mutex                  m_mutex;
        std::atomic_bool                      m_is_serializing{false};
        std::atomic_bool                      m_is_deserializing{false};
        std::string                           m_default_output;

    public:
        virtual ~Serializer() = default;

        void*        Context  = nullptr;

        virtual void SetOnCompleteCallback(on_serializer_complete_fn callback)
        {
            m_complete_callback = callback;
        }

        virtual void SetOnDeserializeCompleteCallback(on_serializer_deserialize_complete_fn callback)
        {
            m_deserialize_complete_callback = callback;
        }

        virtual void SetOnProgressCallback(on_serializer_progress_fn callback)
        {
            m_progress_callback = callback;
        }

        virtual void SetOnErrorCallback(on_serializer_error_fn callback)
        {
            m_error_callback = callback;
        }

        virtual void SetOnLogCallback(on_serializer_log_fn callback)
        {
            m_log_callback = callback;
        }

        virtual void SetDefaultOutput(std::string_view output)
        {
            m_default_output = output;
        }

        virtual bool IsSerializing()
        {
            std::lock_guard l(m_mutex);
            return m_is_serializing;
        }

        virtual void Serialize(const ZEngine::Helpers::Ref<TSerializerData>& data) = 0;
        virtual void Deserialize(std::string_view filename)                        = 0;
    };
} // namespace Tetragrama::Serializers