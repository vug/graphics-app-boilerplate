# Example Projects

# 11 - Rendering Cookbook - Infinite Grid

Implementation of [Chapter5/GL01\_Grid/src/main\.cpp at master · PacktPublishing/3D\-Graphics\-Rendering\-Cookbook](https://github.com/PacktPublishing/3D-Graphics-Rendering-Cookbook/blob/master/Chapter5/GL01_Grid/src/main.cpp) from [3D Graphics Rendering Cookbook \| Packt](https://www.packtpub.com/product/3d-graphics-rendering-cookbook/9781838986193) in graphics-boilerplate-app.

![image](screenshots/11-rendering-cookbook-infinite-grid.png)

* Procedurally generate an "infinite quad" that follows camera's projection onto xz-plane in vertex shader
* In fragment shader draw anti-aliased lines of 3 LODs.
* Uses `fwidth(p2D)` for determining LODs and anti-aliasing a lot. 
* also reduce alpha when grid is far away, both for visual appealing and prevention of Moire patterns at distance.

# 10 - UV-Atlas generation and Lightmaps

## Atlas Generation
After a scene is loaded can generate a UV atlas using https://github.com/jpcy/xatlas.
There is a GUI to tweak XAtlas parameters. 

![image](screenshots/10-uv-atlas-1-Scene-UV-using-mesh-uvs.jpg)

Observed that, if input mesh UVs are not given to XAtlas, generated UVs are too scattered and seams causes visual artifacts

![image](screenshots/10-uv-atlas-2-UV-viz-with-mesh-uvs.jpg)

![image](screenshots/10-uv-atlas-3-UV-viz-without-mesh-uvs.jpg)

## Scene export
Export whole scene as a single OBJ file, file contains all the meshes, where atlas UVs are stored in uv attribute and world positions are stored in position attribute (i.e. "scene UVs" and transforms are "baked" with the export)
Export UV2s of all meshes into a binary custom formatted file.

## Light baking in Blender

* Import OBJ in Blender.
* Adjust lighting: I used "Sky Texture" with dramatic sunrise lighting.
* Bake via [Render Baking — Blender Manual](https://docs.blender.org/manual/en/latest/render/cycles/baking.html)

![image](screenshots/10-uv-atlas-4-Blender-bake.jpg)

## Rendering w/Lightmap

To render the scene with the baked lightmap, once scene is loaded, 
* load UV2s from custom binary file and replace existing UVs
* assign lightmap shader instead of original shaders
* bind lightmap texture from blender

![image](screenshots/10-uv-atlas-5-rendering-lightmap.jpg)

# 09 - LearnOpenGL - ShadowMaps

Implementation of [LearnOpenGL \- Shadow Mapping](https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping) in graphics-app-boilerplate.

![image](screenshots/09-learnopengl-shadowmapping.png)

* First, draws depth buffer from directional light source's view in orthographic projection into an offscreen depth-onlyFramebuffer called shadowmap
* the lighting calculations in the shaders have a shadow logic, which 
  i. calculates the distance of the surface to the light,
  i. compares that to the distance stored in the shadow map (requires vertex positions to be transform into Light-space),
  i. if stored value in closer than that fragment is in shadow

Requires extra care for fragments that are outside of the "light camera"s view plane, z-fighting of a surface with itself (aka shadow acne) etc.


# 08 - LearnOpenGL - Cubemaps

Implementation of [LearnOpenGL \- Cubemaps](https://learnopengl.com/Advanced-OpenGL/Cubemaps) in graphics-app-boilerplate.

Two ways of drawing a skybox via cubemap:
* First draw skybox, then draw scene. Some pixels will be drawn twice
* First draw the scene, store depth, draw skybox after, only for unoccupied pixels

Also introduces a reflective+refractive shader to use the environment map to create glassy refractive and metallic reflective surfaces (still unlit)

![image](screenshots/08-learnopengl-cubemaps.png)

Observe that the "refraction" just displays the skybox on the surface of the object. So, if the container is behind the teapot, it's not visible. This is not true transparency. 

# 07 - Outlines via "Growth"

An alternative to the previous app. This time the outline is done via

* Draw the scene to screen
* Draw highlighted objects offscreen with a solid color shader
* Blit above texture into another offscreen one, via a shader that makes everywhere transparent except the pixels whose colored neighbor count N satisfies 0 < N < 8.
* Blit above texture to screen with blending on

With this technique no artifacts due to scaling up models occurs!

![image](screenshots/07-outlines-via-grow.png)

# 06 LearnOpenGL - Stencil Testing for Outlines

Implementation of [LearnOpenGL \- Stencil testing](https://learnopengl.com/Advanced-OpenGL/Stencil-testing) in graphics-app-boilerplate.
Observe the uneven thickness of the outline.

![image](screenshots/06-learnopengl-stencil-testing.png)

# 05 Sand Automata via Margolus Neighborhoods

Cellular automata for 2D falling sand simulation.

![image](screenshots/05-margolus-automata-sand.png)

See following for explanations

* [Block cellular automaton \- Wikipedia](https://en.wikipedia.org/wiki/Block_cellular_automaton)
* [Reversible cellular automata simulator: help page](https://dmishin.github.io/js-revca/help.html)

# 04 CUDA Mandelbrot

Draws Mandelbrot and Julia sets via CUDA. Can pan and zoom-in.

![image](screenshots/04-cuda-mandelbrot-1.png)

When using 32-bit floats the zoom-in limit is about 10^-6. (height side corresponds an interval of 4.1e-6)
![image](screenshots/04-cuda-mandelbrot-2.png)

Using doubles (64-bit floating points) fixes the discritization/pixellation problem at the same zoom-level
![image](screenshots/04-cuda-mandelbrot-3.png)

With doubles we can go down way further. Here height side corresponds to an interval of 3.1e-14 or something.
![image](screenshots/04-cuda-mandelbrot-4.png)

Enjoy zooming in and out multiple orders of magnitude
![image](screenshots/04-cuda-mandelbrot-5.gif)

# 03 First CUDA app

First CUDA app prepared via CMake. 
Project structure is such that CUDA code is compiled into a static library.
Which later is linked to C++ app code.

Introduction `launchKERNEL_NAME` code organization in `kernels.h` so that C++ does not involve any C++ CUDA extensions such as `<<<n,m>>>` syntax that causes an error to be reported in Visual Studio.

![image](screenshots/03-cuda-start.png)

# 02 Triangle (Without VBO)

Draws The Triangle without any meshes, just shaders

![image](screenshots/02-triangle-without-vbo.png)

# 01 Clear Window

Simplest app, just clear the window with a solid color which can be set via an ImGui ColorPicker

![image](screenshots/01-clear.png)










