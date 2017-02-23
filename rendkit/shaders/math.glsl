float tr(mat2 S) {
  return S[0][0] + S[1][1];
}

vec2 eig(mat2 S) {
  float tr = tr(S);
  float rt = sqrt(tr*tr - 4 * determinant(S));
  return vec2(tr + rt, tr - rt) / 2;
}

