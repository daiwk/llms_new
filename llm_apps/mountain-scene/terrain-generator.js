function generateTerrain() {
    const SIZE = 100;
    const SEGMENTS = 128;
    
    // Generate height map with multiple noise layers
    const heightMap = new Float32Array(SEGMENTS * SEGMENTS);
    for(let y = 0; y < SEGMENTS; y++) {
        for(let x = 0; x < SEGMENTS; x++) {
            let nx = x/SEGMENTS - 0.5;
            let ny = y/SEGMENTS - 0.5;
            
            // Base mountain shape
            let elevation = Math.abs(noise(nx * 3, ny * 3)) * 15;
            
            // Add cliffs using ridged noise
            elevation += Math.pow(Math.abs(noise(nx * 10, ny * 10)), 2) * 5;
            
            // Fine detail
            elevation += noise(nx * 20, ny * 20) * 0.5;
            
            heightMap[y * SEGMENTS + x] = elevation;
        }
    }

    // Create geometry
    const geometry = new THREE.PlaneGeometry(SIZE, SIZE, SEGMENTS-1, SEGMENTS-1);
    geometry.rotateX(-Math.PI/2);
    
    // Set vertex heights
    const vertices = geometry.attributes.position.array;
    for(let i = 0; i < vertices.length; i += 3) {
        vertices[i+1] = heightMap[i/3];
    }
    geometry.computeVertexNormals();

    // Create material with elevation-based coloring
    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        wireframe: false,
        flatShading: false
    });

    // Set vertex colors based on elevation
    const colors = [];
    const maxHeight = Math.max(...heightMap);
    for(let i = 0; i < vertices.length; i += 3) {
        const height = vertices[i+1];
        const color = new THREE.Color();
        color.setHSL(
            0.6 - height * 0.03, // Hue (bluer at lower elevations)
            0.7 - height * 0.03, // Saturation
            0.3 + height * 0.1   // Lightness
        );
        colors.push(color.r, color.g, color.b);
    }
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    return {
        mesh: new THREE.Mesh(geometry, material),
        heightMap: heightMap
    };
}

function createRiver(heightMap) {
    const SEGMENTS = 128;
    const riverGeometry = new THREE.PlaneGeometry(100, 100, SEGMENTS-1, SEGMENTS-1);
    riverGeometry.rotateX(-Math.PI/2);
    
    // Create water surface with animated waves
    const waterMaterial = new THREE.ShaderMaterial({
        uniforms: {
            time: { value: 0 },
            waterColor: { value: new THREE.Color(0.2, 0.5, 0.8) }
        },
        vertexShader: `
            varying vec3 vPos;
            void main() {
                vPos = position;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform vec3 waterColor;
            varying vec3 vPos;
            void main() {
                float ripple = sin(vPos.x * 0.5 + vPos.z * 0.5 + time * 5.0) * 0.1;
                gl_FragColor = vec4(waterColor + vec3(ripple * 0.2), 0.7);
            }
        `,
        transparent: true
    });

    const water = new THREE.Mesh(riverGeometry, waterMaterial);
    water.position.y = 2; // Set water level
    return water;
}

// Simple 2D noise implementation
function noise(x, y) {
    return Math.sin(x * 10.234 + Math.sin(y * 5.678)) * 
           Math.cos(y * 6.789 + Math.sin(x * 9.123)) * 
           Math.sin(x * y * 4.567);
}
