#version 460

layout (points) in;

struct VertexData {
  vec3 worldPosition;
  vec3 worldNormal;
};
in VertexData v[];

uniform mat4 u_ViewFromWorld;
uniform mat4 u_ProjectionFromView;
uniform float u_NormalVizLength = 0.2;

struct GeomData {
  vec4 color;
};
out GeomData g;

layout (line_strip, max_vertices = 2) out;

void main() {
  int ix = 0;
  gl_Position = u_ProjectionFromView * u_ViewFromWorld * vec4(v[ix].worldPosition, 1.0);
  g.color = vec4(0, 1, 1, 1);
  EmitVertex();
  gl_Position = u_ProjectionFromView * u_ViewFromWorld * vec4(v[ix].worldPosition + normalize(v[ix].worldNormal) * u_NormalVizLength, 1.0);
  g.color = vec4(1, 1, 0, 1);
  EmitVertex();
  EndPrimitive();
} 