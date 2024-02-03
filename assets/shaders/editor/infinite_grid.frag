#version 460

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 p2D;

uniform vec3 u_CameraPosition = vec3(0, 0, -5);

out vec4 FragColor;

float log10(float x) {
	return log(x) / log(10.0);
}
float satf(float x) {
	return clamp(x, 0.0, 1.0);
}
vec2 satv(vec2 x) {
	return clamp(x, vec2(0.0), vec2(1.0));
}
float max2(vec2 v) {
	return max(v.x, v.y);
}

// anti-aliased
void gridViaDiscard() {
  const vec2 cellPos = fract(p2D);
  const vec2 dudv = fwidth(p2D);

  const bool hasWidthOfX = cellPos.x < dudv.x || cellPos.y < dudv.y;
  if (hasWidthOfX) {
    float val = 100;
    float alpha = 1 - smoothstep(val * 0.95, val * 1, length(p2D - u_CameraPosition.xz));
    FragColor = vec4(1, 1, 1, alpha);
  } else {
    discard;
  }
}

void gridNoLod() {
  vec2 r = p2D;
  const float cellSize = 1.;
  const vec2 cellPos = mod(p2D, cellSize);
  vec2 dudv = fwidth(p2D);
  float gridSize = 100;

  if(false) {
	  dudv *= 2.0;
    r += dudv / 1;
    // fract(r) == mod(r, 1)
    float a = max2(vec2(1) - abs(satv(mod(r, cellSize) / dudv) * 2 - vec2(1.0)));
	  r -= u_CameraPosition.xz;
	  float opacityFalloff = (1.0 - satf(length(r) / gridSize));
    FragColor = vec4(0.5, 0.5, 0.5, a * opacityFalloff);
  }
  // my solution, have less anti-aliasing at distance actually, especially with white lines
  else {
    const float t = 1; // thickness
    const float vx = 1 - smoothstep(0, dudv.x * t, cellPos.x) + smoothstep(1 - dudv.x * t, 1, cellPos.x);
    const float vy = 1 - smoothstep(0, dudv.y * t, cellPos.y) + smoothstep(1 - dudv.y * t, 1, cellPos.y);
    const float v = vx + vy;

    r -= u_CameraPosition.xz;
    const float opacityFalloff = (1.0 - satf(length(r) / gridSize));

    FragColor = vec4(0.5, 0.5, 0.5, v * opacityFalloff);
  }
}

void gridWithLods() {
  // extents of grid in world coordinates
  float gridSize = 100.;
  // size of one cell
  float gridCellSize = 0.01;
  float gridMinPixelsBetweenCells = 2;
  // color of thin lines
  vec4 gridColorThin = vec4(0.5, 0.5, 0.5, 1.0);
  // color of thick lines (every tenth line)
  vec4 gridColorThick = vec4(1.0, 1.0, 1.0, 1.0);

  vec2 r = p2D;
  //vec2 dudv = fwidth(p2D);
	vec2 dudv = vec2(length(vec2(dFdx(r.x), dFdy(r.x))), length(vec2(dFdx(r.y), dFdy(r.y))));
  vec2 camPos = u_CameraPosition.xz;

	float lodLevel = max(0.0, log10((length(dudv) * gridMinPixelsBetweenCells) / gridCellSize) + 1.0);
	float lodFade = fract(lodLevel);

	// cell sizes for lod0, lod1 and lod2
	float lod0 = gridCellSize * pow(10.0, floor(lodLevel));
	float lod1 = lod0 * 10.0;
	float lod2 = lod1 * 10.0;

	
  //if (floor(abs(r.x / length(dudv))) == 0)
  //  gridColorThick.rgb = vec3(1, 0, 0);

  // thickness: each anti-aliased line covers up to 3 pixels
  float t = 3.0;
  // Update grid coordinates for subsequent alpha calculations (centers each anti-aliased line)
  vec2 dr = t * 0.5 * dudv;

	// calculate absolute distances to cell line centers for each lod and pick max X/Y to get coverage alpha value
	float lod0a = max2( vec2(1.0) - abs(satv(mod(r + dr, lod0) / dudv / t) * 2.0 - vec2(1.0)) );
	float lod1a = max2( vec2(1.0) - abs(satv(mod(r + dr, lod1) / dudv / t) * 2.0 - vec2(1.0)) );
	float lod2a = max2( vec2(1.0) - abs(satv(mod(r + dr, lod2) / dudv / t) * 2.0 - vec2(1.0)) );

	// blend between falloff colors to handle LOD transition
	vec4 c = lod2a > 0.0 ? gridColorThick : lod1a > 0.0 ? mix(gridColorThick, gridColorThin, lodFade) : gridColorThin;

	// calculate opacity falloff based on distance to grid extents
	r -= camPos;
	float opacityFalloff = (1.0 - satf(length(r) / gridSize));

	// blend between LOD level alphas and scale with opacity falloff
	c.a *= (lod2a > 0.0 ? lod2a : lod1a > 0.0 ? lod1a : (lod0a * (1.0-lodFade))) * opacityFalloff;

  FragColor = c;
}

void main() {
  //gridViaDiscard();
  //gridNoLod();
  gridWithLods();
  return;
}