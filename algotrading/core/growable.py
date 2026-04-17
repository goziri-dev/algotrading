import numpy as np


_PY_TO_NP: dict[type, np.dtype] = {
    int:   np.dtype(np.int64),
    float: np.dtype(np.float64),
    bool:  np.dtype(np.bool_),
    complex: np.dtype(np.complex128),
}


class Growable:
    """Base class for fixed-field numpy array containers that grow dynamically.

    Subclasses declare their fields as a list of ``(attr_name, dtype)`` pairs
    and pass them to ``super().__init__``.  The public ``update(**kwargs)`` and
    private ``_increase_capacity()`` methods iterate over those fields so
    subclasses never have to repeat the boilerplate.

    **WARNING!** When capacity is increased the underlying arrays are replaced
    with new ones, so any external references to the old arrays will become stale.
    """

    _DEFAULT_CAPACITY = 10_000

    @classmethod
    def set_default_capacity(cls, capacity: int) -> None:
        """Set the default capacity used by all Growable instances on creation."""
        Growable._DEFAULT_CAPACITY = capacity

    def __init__(self, fields: list[tuple[str, type]]):
        self._capacity = Growable._DEFAULT_CAPACITY
        self._size = 0
        self._fields = list(fields)  # copy — prevents add_field() from mutating class-level _FIELDS
        for name, dtype in fields:
            setattr(self, name, np.empty(self._capacity, dtype=dtype))

    def set_capacity(self, capacity: int) -> None:
        """Pre-allocate arrays to a specific capacity.

        Can be called before or after data has been added.  Capacity cannot
        be reduced below the number of bars/values already stored.

        **WARNING!** Arrays are replaced with new ones, so any external
        references to the old arrays will become stale.
        """
        if capacity < self._size:
            raise ValueError(f"capacity {capacity} is less than current size {self._size}")
        for name, dtype in self._fields:
            new_array = np.empty(capacity, dtype=dtype)
            new_array[:self._size] = getattr(self, name)[:self._size]
            setattr(self, name, new_array)
        self._capacity = capacity

    def _increase_capacity(self):
        new_capacity = self._capacity * 2
        for name, dtype in self._fields:
            new_array = np.empty(new_capacity, dtype=dtype)
            new_array[:self._size] = getattr(self, name)[:self._size]
            setattr(self, name, new_array)
        self._capacity = new_capacity

    def __len__(self) -> int:
        return self._size

    def add_field(self, name: str, dtype: type | np.dtype) -> None:
        """Add a new field at runtime without subclassing.

        ``dtype`` accepts plain Python types (``int``, ``float``, ``bool``,
        ``complex``) or a numpy dtype directly.  The field is accessible via
        ``__getitem__`` using its public name (without the leading underscore).

        Must be called before any data is fed.
        """
        if self._size > 0:
            raise RuntimeError("add_field must be called before any data is fed")
        np_dtype = _PY_TO_NP.get(dtype, dtype) if isinstance(dtype, type) else dtype  # type: ignore[arg-type]
        attr = f"_{name}"
        self._fields.append((attr, np_dtype))
        setattr(self, attr, np.empty(self._capacity, dtype=np_dtype))

    def append(self, **kwargs) -> None:
        """Mutate fields on the last appended row in place.

        Keys must match the private attribute names (including leading underscore).
        Raises ``RuntimeError`` if no rows have been appended yet.
        """
        if self._size == 0:
            raise RuntimeError("No rows to update — call update() first")
        for key, value in kwargs.items():
            getattr(self, key)[self._size - 1] = value

    def update(self, **kwargs):
        """Append one element to each field array.

        Keys must match the field attribute names (including the leading
        underscore).  Capacity is doubled automatically when full.
        """
        if self._size >= self._capacity:
            self._increase_capacity()
        for key, value in kwargs.items():
            getattr(self, key)[self._size] = value
        self._size += 1
