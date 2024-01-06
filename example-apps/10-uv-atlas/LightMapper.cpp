#include "LightMapper.hpp"

#include <Workshop/Assets.hpp>

#include <imgui.h>
#include <stb_image_write.h>

#include <fstream>
#include <iostream>
#include <print>
#include <ranges>

const std::filesystem::path SRC{SOURCE_DIR};

LightMapper::LightMapper() {
	atlas = xatlas::Create();
}

LightMapper::~LightMapper() {
	xatlas::Destroy(atlas);
}

void LightMapper::generateUV2Atlas(ws::Scene& scene) {
	uint32_t numMeshes = static_cast<uint32_t>(scene.renderables.size());
	uint32_t totalVertices = 0;
	uint32_t totalFaces = 0;

	// xatlas::SetProgressCallback(atlas, ProgressCallback, &stopwatch);
	for (auto& r : scene.renderables) {
		xatlas::MeshDecl& meshDecl = meshDeclarations.emplace_back(calcXAtlasMeshDecl(r.get().mesh, r.get().transform));
		assert(meshDecl.indexCount == static_cast<uint32_t>(r.get().mesh.meshData.indices.size()));
		xatlas::AddMeshError error = xatlas::AddMesh(atlas, meshDecl, numMeshes);
		if (error != xatlas::AddMeshError::Success) {
			xatlas::Destroy(atlas);
			std::println("Error adding mesh {}", xatlas::StringForEnum(error));
			throw std::exception("Couldn't convert scene mesh to atlas mesh");
		}
		totalVertices += meshDecl.vertexCount;
		totalFaces += meshDecl.faceCount > 0 ? meshDecl.faceCount : meshDecl.indexCount / 3; // Assume triangles if MeshDecl::faceCount not specified.
	}
	std::println("{} total vertices, {} total faces", totalVertices, totalFaces);

	std::println("Generating atlas");
	// Default ChartOptions (Might be better if there is a defaultChartOptions member
	xatlas::ChartOptions chartOptions;
	chartOptions.useInputMeshUvs = true;
	xatlas::Generate(atlas, chartOptions);
}

void LightMapper::copyUV2sBackToSceneMesh(uint32_t meshIx, ws::Mesh& sceneMesh) const {
	const xatlas::Mesh& mesh = atlas->meshes[meshIx];
	for (size_t ix = 0; ix < mesh.vertexCount; ++ix) {
		const xatlas::Vertex& v = mesh.vertexArray[ix];
		sceneMesh.meshData.vertices[v.xref].texCoord2 = {v.uv[0] / atlas->width, v.uv[1] / atlas->height};
	}
	sceneMesh.uploadData();
}

// should be called after scene is set and atlas is generated
glm::uvec2 LightMapper::getAtlasSize() const {
	assert(atlas != nullptr);
	return glm::vec2{atlas->width, atlas->height};
}

void LightMapper::drawUI(ws::Scene& scene) {
	ImGui::Begin("LightMapper");
	static xatlas::ChartOptions chartOptions;
	ImGui::Text("Chart Options");
	uint32_t n0 = 0, n1 = 1, n5 = 5, n64 = 64, n1000 = 1000;
	ImGui::SliderFloat("Max Chart Area", &chartOptions.maxChartArea, 0, 10);
	ImGui::SliderFloat("Max Chart Boundary Length", &chartOptions.maxBoundaryLength, 0, 50);
	ImGui::SliderFloat("normalDeviationWeight", &chartOptions.normalDeviationWeight, 0, 10);
	ImGui::SliderFloat("roundnessWeight", &chartOptions.roundnessWeight, 0, 10);
	ImGui::SliderFloat("straightnessWeight", &chartOptions.straightnessWeight, 0, 10);
	ImGui::SliderFloat("normalSeamWeight", &chartOptions.normalSeamWeight, 0, 10);
	ImGui::SliderFloat("textureSeamWeight", &chartOptions.textureSeamWeight, 0, 10);
	ImGui::SliderFloat("Max Cost", &chartOptions.maxCost, 0, 10);
	ImGui::SliderScalar("Max Iterations", ImGuiDataType_U32, &chartOptions.maxIterations, &n1, &n5);
	ImGui::Checkbox("Use Input Mesh UVs", &chartOptions.useInputMeshUvs);
	ImGui::Checkbox("Consistent TexCoord Winding", &chartOptions.fixWinding);
	static xatlas::PackOptions packOptions;
	ImGui::Text("Pack Options");
	ImGui::SliderScalar("Max Chart Size", ImGuiDataType_U32, &packOptions.maxChartSize, &n0, &n1000);
	ImGui::SliderScalar("Padding", ImGuiDataType_U32, &packOptions.padding, &n0, &n5);
	ImGui::SliderFloat("Texels per Unit", &packOptions.texelsPerUnit, 0, 64);
	ImGui::SliderScalar("Resolution", ImGuiDataType_U32, &packOptions.resolution, &n0, &n64);
	ImGui::Checkbox("Leave space for bilinear filtering", &packOptions.bilinear);
	ImGui::Checkbox("Align charts to 4x4 blocks", &packOptions.blockAlign);
	ImGui::Checkbox("Brute Force", &packOptions.bruteForce);
	ImGui::Checkbox("Create Image", &packOptions.createImage);
	ImGui::Checkbox("Rotate Charts to Convex Hull Axis", &packOptions.rotateChartsToAxis);
	ImGui::Checkbox("Rotate Charts", &packOptions.rotateCharts);
	if (ImGui::Button("Regenerate UV Atlas and Upload to UV2s")) {
		xatlas::Generate(atlas, chartOptions, packOptions);
		for (uint32_t ix = 0; ix < atlas->meshCount; ++ix)
			copyUV2sBackToSceneMesh(ix, scene.renderables[ix].get().mesh);
	}

	if (ImGui::Button("Save xatlas generated image") && atlas->image)
		stbi_write_png("uv_atlas_xatlas.png", atlas->width, atlas->height, 4, atlas->image, sizeof(uint32_t) * atlas->width);

	if (ImGui::Button("Export whole scene in world-space into single OBJ w/baked UV2")) {
		const char* modelFilename = "baked_scene.obj";
		std::println("Writing '{}'...", modelFilename);
		std::FILE* file;
		fopen_s(&file, modelFilename, "w");
		assert(file != nullptr);
		uint32_t firstVertex = 0;
		for (uint32_t i = 0; i < atlas->meshCount; i++) {
			const xatlas::Mesh& mesh = atlas->meshes[i];
			for (uint32_t v = 0; v < mesh.vertexCount; v++) {
				const xatlas::Vertex& vertex = mesh.vertexArray[v];
				// world position and normals are already stored in MeshDecls
				const float* vertexPosArr = (const float*)meshDeclarations[i].vertexPositionData;
				std::println(file, "v {:g} {:g} {:g}", vertexPosArr[vertex.xref * 3 + 0], vertexPosArr[vertex.xref * 3 + 1], vertexPosArr[vertex.xref * 3 + 2]);
				const float* vertexNormalArr = (const float*)meshDeclarations[i].vertexNormalData;
				std::println(file, "vn {:g} {:g} {:g}", vertexNormalArr[3 * vertex.xref + 0], vertexNormalArr[3 * vertex.xref + 1], vertexNormalArr[3 * vertex.xref + 2]);
				std::println(file, "vt {:g} {:g}", vertex.uv[0] / atlas->width, vertex.uv[1] / atlas->height);
			}
			std::println(file, "o {}", scene.renderables[i].get().name);
			std::println(file, "s off");
			for (uint32_t f = 0; f < mesh.indexCount; f += 3) {
				std::print(file, "f ");
				for (uint32_t j = 0; j < 3; j++) {
					const uint32_t index = firstVertex + mesh.indexArray[f + j] + 1;  // 1-indexed
					std::print(file, "{:d}/{:d}/{:d}{:c}", index, index, index, j == 2 ? '\n' : ' ');
				}
			}
			firstVertex += mesh.vertexCount;
		}
		std::fclose(file);    
	}

	if (ImGui::Button("Save UV2s")) {
		std::filesystem::path uvFile = SRC / "uv2s.dat";
		std::ofstream out(uvFile.string().c_str(), std::ios::binary);
		assert(out.is_open());
		uint32_t numAtlasMeshes = atlas->meshCount;
		out.write(reinterpret_cast<char*>(&numAtlasMeshes), sizeof(uint32_t));
		for (uint32_t i = 0; i < numAtlasMeshes; i++) {
			const ws::RenderableObject& r = scene.renderables[i];
			const xatlas::Mesh& atlasMesh = atlas->meshes[i];
			size_t objNameLength = r.name.length();
			out.write(reinterpret_cast<char*>(&objNameLength), sizeof(uint32_t));
			std::string objName = r.name;
			out.write(reinterpret_cast<const char*>(objName.c_str()), sizeof(char) * objNameLength);
			uint32_t numVertices = atlasMesh.vertexCount;
			out.write(reinterpret_cast<char*>(&numVertices), sizeof(uint32_t));
			for (uint32_t vIx = 0; vIx < atlasMesh.vertexCount; vIx++) {
				xatlas::Vertex& atlasVertex = atlasMesh.vertexArray[vIx];
				float u = atlasVertex.uv[0] / atlas->width;
				float v = atlasVertex.uv[1] / atlas->height;
				out.write(reinterpret_cast<char*>(&u), sizeof(float));
				out.write(reinterpret_cast<char*>(&v), sizeof(float));
			}
		}
	}

	if (ImGui::Button("Read UV2s")) {
		std::filesystem::path uvFile = SRC / "uv2s.dat";
		std::ifstream in(uvFile.string().c_str(), std::ios::binary);
		assert(in.is_open());
		uint32_t numMeshesInDat;
		in.read(reinterpret_cast<char*>(&numMeshesInDat), sizeof(uint32_t));
		std::println("numMeshes {}", numMeshesInDat);
		for (uint32_t i = 0; i < numMeshesInDat; i++) {
			uint32_t objNameLength;
			in.read(reinterpret_cast<char*>(&objNameLength), sizeof(uint32_t));
			std::println("objNameLength {}", objNameLength);
			char* objNamePtr = new char[objNameLength + 1];
			objNamePtr[objNameLength] = '\0';
			in.read(reinterpret_cast<char*>(objNamePtr), sizeof(char) * objNameLength);
			std::string objName{objNamePtr};
			std::println("objName {}", objName);
			uint32_t numVertices;
			in.read(reinterpret_cast<char*>(&numVertices), sizeof(uint32_t));
			std::println("numVertices {}", numVertices);
			const xatlas::Mesh& atlasMesh = atlas->meshes[i];
			ws::Mesh& mesh = scene.renderables[i].get().mesh;
			std::vector<ws::DefaultVertex>& vertices = mesh.meshData.vertices;
			for (uint32_t vIx = 0; vIx < numVertices; vIx++) {
				uint32_t ix = atlasMesh.vertexArray[vIx].xref;
				glm::vec2 uv2;
				in.read(reinterpret_cast<char*>(&uv2), sizeof(glm::vec2));
				vertices[ix].texCoord2 = uv2;
				//std::print("({:.3f},{:.3f}) ", uv2.x, uv2.y);
			}
			mesh.uploadData();
			std::println("");
		}
	}

	ImGui::Text("Atlas Info");
	ImGui::Text("Size: (%d, %d)", atlas->width, atlas->height);
	ImGui::Text("# meshes: %d", atlas->meshCount);
	ImGui::Text("# atlases: %d", atlas->atlasCount);
	ImGui::Text("# charts: %d", atlas->chartCount);
	ImGui::Text("texelsPerUnit: %f", atlas->texelsPerUnit);

	for (uint32_t i = 0; i < atlas->atlasCount; i++)
		ImGui::Text("Atlas utilization: atlas[%d]: %.2f", i, atlas->utilization[i] * 100.0f);
	const ImVec2 tableOuterSize{0.f, 200.f};
	static int meshIx = 0;
	ImGui::SliderInt("Atlas Mesh Ix", &meshIx, 0, atlas->meshCount - 1);
	ImGui::Text("Obj: %s", scene.renderables[meshIx].get().name.c_str());
	if (ImGui::BeginTable("Vertices", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollY, tableOuterSize)) {
		ImGui::TableSetupColumn("origIx");
		ImGui::TableSetupColumn("atlasIx");
		ImGui::TableSetupColumn("texCoord");
		ImGui::TableSetupColumn("uv");
		ImGui::TableSetupColumn("atlas");
		ImGui::TableSetupColumn("chart");
		ImGui::TableHeadersRow();
		for (size_t ix = 0; ix < atlas->meshes[meshIx].vertexCount; ++ix) {
			const xatlas::Vertex& v = atlas->meshes[meshIx].vertexArray[ix];
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::Text("%3d", v.xref);        
			ImGui::TableNextColumn();
			ImGui::Text("%3d", ix);
			ImGui::TableNextColumn();
			ImGui::Text("(%6.1f, %6.1f)", v.uv[0], v.uv[1]);
			ImGui::TableNextColumn();
			ImGui::Text("(%1.3f, %1.3f)", v.uv[0] / atlas->width, v.uv[1] / atlas->height);
			ImGui::TableNextColumn();
			ImGui::Text("%d", v.atlasIndex);
			ImGui::TableNextColumn();
			ImGui::Text("%d", v.chartIndex);
		}
		ImGui::EndTable();
	}

	ImGui::End();
}

xatlas::MeshDecl LightMapper::calcXAtlasMeshDecl(const ws::Mesh& wsMesh, const ws::Transform& transform) {
	xatlas::MeshDecl meshDecl;
    
	size_t numVertices = wsMesh.meshData.vertices.size();
	float* positions = new float[numVertices * 3];
	float* normals = new float[numVertices * 3];
	float* texCoords = new float[numVertices * 2];
	for (const auto& [ix, v] : wsMesh.meshData.vertices | std::ranges::views::enumerate) {
		const bool useWorldSpace = true;
		if (useWorldSpace) {
			glm::vec3 worldPos = transform.getWorldFromObjectMatrix() * glm::vec4(v.position, 1);
			positions[3 * ix + 0] = worldPos.x;
			positions[3 * ix + 1] = worldPos.y;
			positions[3 * ix + 2] = worldPos.z;
			glm::vec3 worldNormal = glm::normalize(glm::transpose(glm::inverse(transform.getWorldFromObjectMatrix())) * glm::vec4(v.normal, 1));
			normals[3 * ix + 0] = worldNormal.x;
			normals[3 * ix + 1] = worldNormal.y;
			normals[3 * ix + 2] = worldNormal.z;      
		} else {
			positions[3 * ix + 0] = v.position.x;
			positions[3 * ix + 1] = v.position.y;
			positions[3 * ix + 2] = v.position.z;
			normals[3 * ix + 0] = v.normal.x;
			normals[3 * ix + 1] = v.normal.y;
			normals[3 * ix + 2] = v.normal.z;      
		}
		texCoords[2 * ix + 0] = v.texCoord.x;
		texCoords[2 * ix + 1] = v.texCoord.y;
	}
	meshDecl.vertexCount = static_cast<uint32_t>(wsMesh.meshData.vertices.size());
	meshDecl.vertexPositionData = positions;
	meshDecl.vertexPositionStride = sizeof(float) * 3;
	meshDecl.vertexNormalData = normals;
	meshDecl.vertexNormalStride = sizeof(float) * 3;
	meshDecl.vertexUvData = texCoords;
	meshDecl.vertexUvStride = sizeof(float) * 2;
	meshDecl.indexCount = static_cast<uint32_t>(wsMesh.meshData.indices.size());
	meshDecl.indexData = wsMesh.meshData.indices.data();
	meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
	return meshDecl;
}
