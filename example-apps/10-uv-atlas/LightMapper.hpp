#pragma once

#include <Workshop/Scene.hpp>
#include <Workshop/Transform.hpp>

#include <xatlas.h>
#include <glm/vec2.hpp>

class LightMapper {
public:
	LightMapper();
	~LightMapper();

	void generateUV2Atlas(ws::Scene& scene);
	void copyUV2sBackToSceneMesh(uint32_t meshIx, ws::Mesh& sceneMesh) const;
	glm::uvec2 getAtlasSize() const;
	void drawUI(ws::Scene& scene);
private:
	xatlas::Atlas* atlas;
	std::vector<xatlas::MeshDecl> meshDeclarations;

	xatlas::MeshDecl calcXAtlasMeshDecl(const ws::Mesh& wsMesh, const ws::Transform& transform);
};