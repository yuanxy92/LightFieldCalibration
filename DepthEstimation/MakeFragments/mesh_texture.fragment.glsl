// opengl shader file
// used to render textures
// author: Shane Yuan
// date: Apr 15, 2018
//
#version 440 core
// Interpolated values from the vertex shaders
in vec2 UV;
// Ouput data
out vec4 colorRGBA;
// Values that stay constant for the whole mesh.
uniform sampler2D myTextureSampler;
// main function
void main(){
	colorRGBA = texture(myTextureSampler, UV);
}