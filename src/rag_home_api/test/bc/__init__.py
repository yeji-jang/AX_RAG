"""일반 전월세 RAG의 팀 서비스 진입점. import만으로 API·DB·모델을 실행하지 않는다."""

__all__ = ['CachedPipeline', 'ResponseCache']


def __getattr__(name):
    if name in __all__:
        from .cached_pipeline import CachedPipeline, ResponseCache
        return {'CachedPipeline': CachedPipeline, 'ResponseCache': ResponseCache}[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
