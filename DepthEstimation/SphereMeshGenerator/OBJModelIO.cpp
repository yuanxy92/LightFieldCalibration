/**
@brief class for *.obj file IO
@author Shane Yuan
@date May 26, 2018
*/

#include "OBJModelIO.h"

/************************************************************************/
/*                     triangle mesh with texture                       */
/************************************************************************/
three::TexTriangleMesh::TexTriangleMesh() {}
three::TexTriangleMesh::~TexTriangleMesh() {}

three::TexTriangleMesh & three::TexTriangleMesh::operator+=(const three::TexTriangleMesh &mesh) {
	if (mesh.IsEmpty()) return (*this);
	size_t old_vert_num = vertices_.size();
	size_t add_vert_num = mesh.vertices_.size();
	size_t new_vert_num = old_vert_num + add_vert_num;
	size_t old_tri_num = triangles_.size();
	size_t add_tri_num = mesh.triangles_.size();
	size_t new_tri_num = old_tri_num + add_tri_num;
	if ((!HasVertices() || HasVertexNormals()) && mesh.HasVertexNormals()) {
		vertex_normals_.resize(new_vert_num);
		for (size_t i = 0; i < add_vert_num; i++)
			vertex_normals_[old_vert_num + i] = mesh.vertex_normals_[i];
	}
	else {
		vertex_normals_.clear();
	}
	if ((!HasVertices() || HasVertexColors()) && mesh.HasVertexColors()) {
		vertex_colors_.resize(new_vert_num);
		for (size_t i = 0; i < add_vert_num; i++)
			vertex_colors_[old_vert_num + i] = mesh.vertex_colors_[i];
	}
	else {
		vertex_colors_.clear();
	}
	vertices_.resize(new_vert_num);
	for (size_t i = 0; i < add_vert_num; i++)
		vertices_[old_vert_num + i] = mesh.vertices_[i];
	vertex_uvs_.resize(new_vert_num);
	for (size_t i = 0; i < add_vert_num; i++) 
		vertex_uvs_[old_vert_num + i] = mesh.vertex_uvs_[i];

	if ((!HasTriangles() || HasTriangleNormals()) &&
		mesh.HasTriangleNormals()) {
		triangle_normals_.resize(new_tri_num);
		for (size_t i = 0; i < add_tri_num; i++)
			triangle_normals_[old_tri_num + i] = mesh.triangle_normals_[i];
	}
	else {
		triangle_normals_.clear();
	}
	triangles_.resize(triangles_.size() + mesh.triangles_.size());
	Eigen::Vector3i index_shift((int)old_vert_num, (int)old_vert_num,
		(int)old_vert_num);
	for (size_t i = 0; i < add_tri_num; i++) {
		triangles_[old_tri_num + i] = mesh.triangles_[i] + index_shift;
	}
	return (*this);
}

three::TexTriangleMesh three::TexTriangleMesh::operator+(
	const three::TexTriangleMesh &mesh) const {
	return (three::TexTriangleMesh(*this) += mesh);
}

/************************************************************************/
/*                         OBJ model IO class                           */
/************************************************************************/
three::OBJModelIO::OBJModelIO() {}
three::OBJModelIO::~OBJModelIO() {}

/**
@brief load obj file
@param std::string filename: input obj filename
@param std::shared_ptr<TexTriangleMesh> & texTriMesh: output triangle mesh
with texture
@return int
*/
int three::OBJModelIO::load(std::string filename,
	std::shared_ptr<three::TexTriangleMesh> & texTriMesh) {

	return 0;
}

/**
@brief save obj model with texture
@param std::string dir: output dir to save the obj file
@param std::string filename: output obj filename
@param std::shared_ptr<TexTriangleMesh> texTriMesh: triangle mesh with texture
@return int
*/
int three::OBJModelIO::save(std::string dir,
	std::string filename, 
	std::shared_ptr<three::TexTriangleMesh> texTriMesh) {
	
	// write obj file
	std::fstream fs(cv::format("%s\\%s.obj", dir.c_str(),
		filename.c_str()), std::ios::out);
	// write file head, first 4 lines
	fs << "# Produced by OBJModelIO class exporter" << std::endl;
	fs << "# Author: Shane Yuan     " << std::endl;
	fs << "# Email: yuanxy92@gmail.com     " << std::endl;
	fs << "# Date: June 2, 2018     " << std::endl;
	fs << "#" << std::endl;
	// write line to denote the material file
	fs << cv::format("mtllib %s.mtl", filename.c_str()) << std::endl;
	// write v
	for (size_t i = 0; i < texTriMesh->vertices_.size(); i++) {
		fs << cv::format("v %lf %lf %lf", texTriMesh->vertices_[i](0),
			texTriMesh->vertices_[i](1), texTriMesh->vertices_[i](2))
			<< std::endl;
	}
	fs << cv::format("# %ld vertices", texTriMesh->vertices_.size()) << std::endl;
	// write vn
	for (size_t i = 0; i < texTriMesh->vertex_normals_.size(); i++) {
		fs << cv::format("vn %lf %lf %lf", texTriMesh->vertex_normals_[i](0),
			texTriMesh->vertex_normals_[i](1), texTriMesh->vertex_normals_[i](2))
			<< std::endl;
	}
	fs << cv::format("# %ld normals", texTriMesh->vertex_normals_.size()) << std::endl;
	// write vt
	for (size_t i = 0; i < texTriMesh->vertex_uvs_.size(); i++) {
		fs << cv::format("vt %lf %lf", texTriMesh->vertex_uvs_[i](0),
			texTriMesh->vertex_uvs_[i](1)) << std::endl;
	}
	fs << cv::format("# %ld texture verticies", texTriMesh->vertex_uvs_.size()) << std::endl;
	fs << "####" << std::endl;
	fs << "####" << std::endl;
	fs << "####" << std::endl;
	// write use material
	fs << "usemtl material_0" << std::endl;
	// write triangles
	//for (size_t i = 0; i < 100; i++) {
	for (size_t i = 0; i < texTriMesh->triangles_.size(); i++) {
		Eigen::Vector3i triangle = texTriMesh->triangles_[i];
		fs << cv::format("f %ld/%ld/%ld %ld/%ld/%ld %ld/%ld/%ld", 
			triangle(0) + 1, triangle(0) + 1, triangle(0) + 1, 
			triangle(1) + 1, triangle(1) + 1, triangle(1) + 1, 
			triangle(2) + 1, triangle(2) + 1, triangle(2) + 1) << std::endl;
	}
	fs << cv::format("# %ld faces", texTriMesh->triangles_.size()) << std::endl;
	fs.close();

	// write mtl file
	fs.open(cv::format("%s\\%s.mtl", dir.c_str(), filename.c_str()), std::ios::out);
	// write file head, first 4 lines
	std::string imagename = cv::format("%s.png", filename.c_str());
	fs << "# Produced by OBJModelIO class exporter" << std::endl;
	fs << "# Author: Shane Yuan     " << std::endl;
	fs << "# Email: yuanxy92@gmail.com     " << std::endl;
	fs << "# Date: June 2, 2018     " << std::endl;
	fs << "#" << std::endl;
	fs << "newmtl material_0" << std::endl;
	fs << "Ka 1.00000 1.00000 1.00000" << std::endl;
	fs << "Kd 1.00000 1.00000 1.00000" << std::endl;
	fs << "Ks 1.00000 1.00000 1.00000" << std::endl;
	fs << "Tr 1.00000" << std::endl;
	fs << "illum 2" << std::endl;
	fs << "Ns 0.00000" << std::endl;
	fs << cv::format("map_Kd %s", imagename.c_str()) << std::endl;
	fs << cv::format("map_Ka %s", imagename.c_str()) << std::endl;
	fs << "# " << std::endl;
	fs << "# EOF" << std::endl;
	fs.close();

	// write image into file
	std::string imgname_full = cv::format("%s/%s", dir.c_str(), imagename.c_str());
	cv::imwrite(imgname_full, texTriMesh->texture_);
	return 0;
}
