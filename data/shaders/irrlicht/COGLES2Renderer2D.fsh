// Copyright (C) 2009-2010 Amundis
// Heavily based on the OpenGL driver implemented by Nikolaus Gebhardt
// and OpenGL ES driver implemented by Christian Stehno
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in Irrlicht.h

precision mediump float;

// float rather than bool: GXP has no bool type, so vitaGL reports
// bool uniforms as GL_FLOAT and irrlicht's int upload path then
// silently refuses them (default: status = false).
uniform float uUseTexture;
uniform sampler2D uTextureUnit;

varying vec4 vVertexColor;
varying vec2 vTexCoord;

void main(void)
{
	vec4 Color = vVertexColor;

	if(uUseTexture > 0.5)
		Color *= texture2D(uTextureUnit, vTexCoord);
	
	gl_FragColor = Color;
}
