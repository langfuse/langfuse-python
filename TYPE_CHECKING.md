# Type Checking with mypy

This project uses [mypy](https://mypy.readthedocs.io/) for static type checking to ensure code quality and catch potential bugs before runtime.

## Status

✅ **100% Type Coverage** - All mypy errors have been resolved

## Configuration

Type checking is configured in `pyproject.toml` under the `[tool.mypy]` section with strict settings:

- `disallow_untyped_defs = true` - All functions must have type annotations
- `disallow_incomplete_defs = true` - All function signatures must be complete
- `warn_return_any = true` - Warns when returning `Any` type
- `strict_equality = true` - Strict equality checking
- Performance optimizations enabled for CI

## Running Type Checks

### Locally

```bash
# Run mypy on the entire langfuse package
poetry run mypy langfuse

# Run with concise output
poetry run mypy langfuse --no-error-summary

# Run on specific files
poetry run mypy langfuse/client.py
```

### Pre-commit Hooks

Type checking runs automatically on all staged Python files when committing:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run mypy --all-files
```

### CI Pipeline

Type checking runs automatically in CI for all pull requests in a dedicated `type-checking` job that:

- Uses Python 3.9 (minimum supported version)
- Caches dependencies and mypy cache for performance
- Fails the build if any type errors are found

## Adding New Code

When adding new code, ensure:

1. **All functions have type annotations**:
   ```python
   def process_data(items: List[str]) -> Dict[str, int]:
       return {}
   ```

2. **Use proper Optional types**:
   ```python
   def get_user(user_id: Optional[str] = None) -> Optional[User]:
       return None
   ```

3. **Import types when needed**:
   ```python
   from typing import Any, Dict, List, Optional, Union
   ```

4. **Use type ignores sparingly**:
   ```python
   # Only when absolutely necessary
   value = external_lib.get_value()  # type: ignore[attr-defined]
   ```

## External Dependencies

External libraries without type stubs are configured to ignore missing imports in the `[[tool.mypy.overrides]]` section.

## Performance

- **Incremental checking**: Only changed files are re-checked
- **SQLite cache**: Persistent cache for faster subsequent runs
- **CI caching**: GitHub Actions caches mypy cache between runs

## Troubleshooting

### Common Issues

1. **Import errors**: Add missing imports to the overrides section
2. **Any types**: Add proper type annotations instead of relying on `Any`
3. **Optional handling**: Use proper `Optional[T]` or `Union[T, None]` types

### Getting Help

- [mypy documentation](https://mypy.readthedocs.io/)
- [Python typing module](https://docs.python.org/3/library/typing.html)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)