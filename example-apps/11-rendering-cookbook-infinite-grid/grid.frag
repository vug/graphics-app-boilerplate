#version 460

//layout(location = 0) in vec2 uv;
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

void main() {
  //FragColor = vec4(1, 0, 0, 1);

  //FragColor = vec4(uv.x, uv.y, 0, 1);

  vec2 dP2D = fwidth(p2D) * 1;

  // params
  // extents of grid in world coordinates
  float gridSize = 100.;
  // size of one cell
  float gridCellSize = 0.01;
  float gridMinPixelsBetweenCells = 2;
  // color of thin lines
  vec4 gridColorThin = vec4(0.5, 0.5, 0.5, 1.0);
  // color of thick lines (every tenth line)
  vec4 gridColorThick = vec4(1.0, 1.0, 1.0, 1.0);

  vec2 uv = p2D;
  //vec2 dudv = dP2D;
	vec2 dudv = vec2(length(vec2(dFdx(uv.x), dFdy(uv.x))), length(vec2(dFdx(uv.y), dFdy(uv.y))));
  vec2 camPos = u_CameraPosition.xz;


	float lodLevel = max(0.0, log10((length(dudv) * gridMinPixelsBetweenCells) / gridCellSize) + 1.0);
	float lodFade = fract(lodLevel);

	// cell sizes for lod0, lod1 and lod2
	float lod0 = gridCellSize * pow(10.0, floor(lodLevel));
	float lod1 = lod0 * 10.0;
	float lod2 = lod1 * 10.0;

	// each anti-aliased line covers up to 4 pixels
	dudv *= 4.0;

	// Update grid coordinates for subsequent alpha calculations (centers each anti-aliased line)
  uv += dudv / 2.0F;

	// calculate absolute distances to cell line centers for each lod and pick max X/Y to get coverage alpha value
	float lod0a = max2( vec2(1.0) - abs(satv(mod(uv, lod0) / dudv) * 2.0 - vec2(1.0)) );
	float lod1a = max2( vec2(1.0) - abs(satv(mod(uv, lod1) / dudv) * 2.0 - vec2(1.0)) );
	float lod2a = max2( vec2(1.0) - abs(satv(mod(uv, lod2) / dudv) * 2.0 - vec2(1.0)) );

	uv -= camPos;

	// blend between falloff colors to handle LOD transition
	vec4 c = lod2a > 0.0 ? gridColorThick : lod1a > 0.0 ? mix(gridColorThick, gridColorThin, lodFade) : gridColorThin;

	// calculate opacity falloff based on distance to grid extents
	float opacityFalloff = (1.0 - satf(length(uv) / gridSize));

	// blend between LOD level alphas and scale with opacity falloff
	c.a *= (lod2a > 0.0 ? lod2a : lod1a > 0.0 ? lod1a : (lod0a * (1.0-lodFade))) * opacityFalloff;

  const vec2 cellPos = fract(p2D);
  //FragColor = vec4(cellPos, 0, 1);


  float v1 = 1 - smoothstep(0, dP2D.x * 2, cellPos.x);
  float s = 1.;
  v1 += smoothstep(1 - dP2D.x * s, 1, cellPos.x);

  float v2 = 1 - smoothstep(0, dP2D.y * 2, cellPos.y);
  v2 += smoothstep(1 - dP2D.y * s, 1, cellPos.y);
  float v = v1 + v2;

  float val = 75;
  float alpha = 1 - smoothstep(val * 0.95, val * 1, length(p2D - u_CameraPosition.xz));
  FragColor = vec4(v, v, v, v * alpha);

  FragColor = c;
  //FragColor = vec4(vec3(lod2), 1);

  /*
  const bool hasWidthOfX = cellPos.x < dP2D.x || cellPos.y < dP2D.y;
  if (hasWidthOfX) {
    float val = 100;
    float alpha = 1 - smoothstep(val * 0.95, val * 1, length(p2D - u_CameraPosition.xz));
    FragColor = vec4(1, 1, 1, alpha);
  } else {
    discard;
  }
  */
}