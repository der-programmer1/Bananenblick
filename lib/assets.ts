/** Works at github.io/ as well as github.io/any-repository/. */
export function assetPathFor(base: string, asset: string): string {
  const prefix = base.endsWith("/") ? base : `${base}/`;
  return prefix + asset.replace(/^\/+/, "");
}
export function assetPath(asset: string): string {
  return assetPathFor(import.meta.env?.BASE_URL || "/", asset);
}
