// A fixed-capacity ring of equal-length `Uint8Array` columns, backing the
// micro-Doppler waterfall history. Pre-allocated — no per-frame garbage.

export class ColumnRing {
  private readonly buffer: Uint8Array;
  private writeIndex = 0;
  private filled = 0;

  constructor(
    private readonly capacity: number,
    private readonly columnLength: number,
  ) {
    if (capacity < 1 || columnLength < 1) {
      throw new Error('ColumnRing requires positive capacity and columnLength');
    }
    this.buffer = new Uint8Array(capacity * columnLength);
  }

  /** Append a column, evicting the oldest when full. */
  push(column: Uint8Array): void {
    const offset = this.writeIndex * this.columnLength;
    if (column.length === this.columnLength) {
      this.buffer.set(column, offset);
    } else {
      // Resample by nearest-neighbour into the fixed column length.
      for (let i = 0; i < this.columnLength; i++) {
        const src = Math.min(
          column.length - 1,
          Math.round((i / Math.max(1, this.columnLength - 1)) * (column.length - 1)),
        );
        this.buffer[offset + i] = column[src] ?? 0;
      }
    }
    this.writeIndex = (this.writeIndex + 1) % this.capacity;
    if (this.filled < this.capacity) {
      this.filled += 1;
    }
  }

  /** Number of stored columns. */
  get length(): number {
    return this.filled;
  }

  /** Column at `age` (0 = newest), or `null` when out of range. */
  at(age: number): Uint8Array | null {
    if (age < 0 || age >= this.filled) {
      return null;
    }
    const index = (this.writeIndex - 1 - age + this.capacity * 2) % this.capacity;
    const offset = index * this.columnLength;
    return this.buffer.subarray(offset, offset + this.columnLength);
  }

  clear(): void {
    this.writeIndex = 0;
    this.filled = 0;
  }
}
