/** Shared named empty-state element for radar panels. */
export default function RadarEmptyState({ message }: { message: string }) {
  return (
    <div className="radar-empty" data-testid="radar-empty">
      {message}
    </div>
  );
}
