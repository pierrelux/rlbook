import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
import { Play, Pause, RotateCcw, Leaf, Fish as FishIcon, Zap } from 'lucide-react';
import * as THREE from 'three';

// Terrain component with height displacement
const Terrain = () => {
  const meshRef = useRef();
  
  const { geometry, material } = useMemo(() => {
    const geo = new THREE.PlaneGeometry(20, 20, 64, 64);
    
    // Generate height displacement for mountainous terrain
    const positions = geo.attributes.position.array;
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i];
      const y = positions[i + 1];
      
      // Create mountain ridges and valleys
      const height = Math.sin(x * 0.3) * Math.cos(y * 0.2) * 2 + 
                    Math.sin(x * 0.1) * 1.5 + 
                    Math.random() * 0.3;
      positions[i + 2] = height;
    }
    geo.computeVertexNormals();
    
    const mat = new THREE.MeshLambertMaterial({
      color: '#10b981',
      transparent: true,
      opacity: 0.8
    });
    
    return { geometry: geo, material: mat };
  }, []);

  return (
    <mesh ref={meshRef} geometry={geometry} material={material} rotation-x={-Math.PI / 2} position={[0, -1, 0]} />
  );
};

// Water component with animated waves
const Water = ({ level = 0 }) => {
  const meshRef = useRef();
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.material.normalScale.set(
        0.1 + Math.sin(state.clock.elapsedTime) * 0.05,
        0.1 + Math.cos(state.clock.elapsedTime * 0.7) * 0.05
      );
    }
  });

  const material = useMemo(() => {
    const mat = new THREE.MeshPhongMaterial({
      color: '#3b82f6',
      transparent: true,
      opacity: 0.7,
      normalMap: new THREE.TextureLoader().load('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='), // 1x1 blue pixel as fallback
    });
    mat.normalScale = new THREE.Vector2(0.1, 0.1);
    return mat;
  }, []);

  return (
    <mesh ref={meshRef} material={material} rotation-x={-Math.PI / 2} position={[0, level, 0]}>
      <planeGeometry args={[18, 18, 32, 32]} />
    </mesh>
  );
};

// Dam component with ecological status
const Dam = ({ position, name, ecologicalImpact, volume, maxVolume, power, onClick }) => {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  
  const impactColor = ecologicalImpact === 'high' ? '#ef4444' : 
                     ecologicalImpact === 'moderate' ? '#f59e0b' : '#10b981';
  
  const fillLevel = volume / maxVolume;
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.scale.y = 1 + (hovered ? 0.1 : 0);
    }
  });

  return (
    <group position={position}>
      {/* Dam structure */}
      <mesh
        ref={meshRef}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <boxGeometry args={[1, 2, 0.3]} />
        <meshLambertMaterial color="#6b7280" />
      </mesh>
      
      {/* Ecological status indicator */}
      <mesh position={[0, 1.2, 0]}>
        <sphereGeometry args={[0.2]} />
        <meshEmissiveMaterial color={impactColor} />
      </mesh>
      
      {/* Reservoir water */}
      <mesh position={[-2, 0, 0]} rotation-x={-Math.PI / 2}>
        <planeGeometry args={[3, 2]} />
        <meshPhongMaterial 
          color="#3b82f6" 
          transparent 
          opacity={0.3 + fillLevel * 0.4} 
        />
      </mesh>
      
      {/* Power generation indicator */}
      {power > 10 && (
        <mesh position={[0.8, 1, 0]}>
          <sphereGeometry args={[0.15]} />
          <meshEmissiveMaterial color="#fbbf24" />
        </mesh>
      )}
      
      {/* Dam label */}
      <Html position={[0, 2.5, 0]} center>
        <div className="bg-white/90 px-2 py-1 rounded text-xs font-semibold text-gray-800 shadow-lg">
          {name}
        </div>
      </Html>
      
      {hovered && (
        <Html position={[0, -1, 0]} center>
          <div className="bg-black/80 text-white px-3 py-2 rounded-lg text-sm">
            <div>Volume: {volume.toFixed(1)}/{maxVolume} hm³</div>
            <div>Power: {power.toFixed(0)} MW</div>
            <div>Impact: {ecologicalImpact}</div>
          </div>
        </Html>
      )}
    </group>
  );
};

// Fish component with swimming animation
const Fish = ({ position, health, moving, blocked }) => {
  const meshRef = useRef();
  const [swimOffset] = useState(() => Math.random() * Math.PI * 2);
  
  const healthColor = health > 0.7 ? '#10b981' : 
                     health > 0.4 ? '#f59e0b' : '#ef4444';
  
  useFrame((state) => {
    if (meshRef.current && moving) {
      const time = state.clock.elapsedTime;
      meshRef.current.position.x = position[0] + Math.sin(time + swimOffset) * 0.5;
      meshRef.current.position.z = position[2] + Math.cos(time * 0.7 + swimOffset) * 0.3;
      meshRef.current.rotation.y = Math.sin(time + swimOffset) * 0.3;
    }
  });

  return (
    <group>
      <mesh 
        ref={meshRef} 
        position={position}
        scale={blocked ? [0.7, 0.7, 0.7] : [1, 1, 1]}
      >
        <sphereGeometry args={[0.15, 8, 6]} />
        <meshLambertMaterial color={healthColor} opacity={health} transparent />
      </mesh>
      
      {blocked && (
        <mesh position={[position[0], position[1] + 0.5, position[2]]}>
          <sphereGeometry args={[0.1]} />
          <meshEmissiveMaterial color="#ef4444" />
        </mesh>
      )}
    </group>
  );
};

// Vegetation component
const Vegetation = ({ position, health, waterAccess }) => {
  const meshRef = useRef();
  
  const healthColor = health > 0.7 ? '#22c55e' : 
                     health > 0.4 ? '#eab308' : '#dc2626';
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime + position[0]) * 0.1;
    }
  });

  return (
    <group position={position}>
      <mesh ref={meshRef}>
        <coneGeometry args={[0.2, 0.8, 6]} />
        <meshLambertMaterial color={healthColor} opacity={health} transparent />
      </mesh>
      
      {!waterAccess && (
        <mesh position={[0, 0.5, 0]}>
          <ringGeometry args={[0.3, 0.4]} />
          <meshBasicMaterial color="#dc2626" transparent opacity={0.5} />
        </mesh>
      )}
    </group>
  );
};

// Flow particles between dams
const FlowParticles = ({ from, to, intensity }) => {
  const particlesRef = useRef();
  const particleCount = Math.ceil(intensity / 10);
  
  const positions = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const t = i / particleCount;
      pos[i * 3] = from[0] + (to[0] - from[0]) * t;
      pos[i * 3 + 1] = from[1] + (to[1] - from[1]) * t + Math.sin(t * Math.PI) * 0.5;
      pos[i * 3 + 2] = from[2] + (to[2] - from[2]) * t;
    }
    return pos;
  }, [from, to, particleCount]);
  
  useFrame((state) => {
    if (particlesRef.current) {
      const time = state.clock.elapsedTime;
      const positions = particlesRef.current.geometry.attributes.position.array;
      
      for (let i = 0; i < particleCount; i++) {
        const offset = (time + i * 0.1) % 1;
        positions[i * 3] = from[0] + (to[0] - from[0]) * offset;
        positions[i * 3 + 1] = from[1] + Math.sin(offset * Math.PI) * 0.3;
        positions[i * 3 + 2] = from[2] + (to[2] - from[2]) * offset;
      }
      particlesRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial color="#60a5fa" size={0.1} />
    </points>
  );
};

// Main 3D scene component
const HydroScene = ({ dams, ecosystemElements, selectedPolicy, showEcological }) => {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} castShadow />
      <directionalLight position={[-10, 5, -5]} intensity={0.3} />
      
      {/* Terrain */}
      <Terrain />
      
      {/* Water */}
      <Water level={-0.5} />
      
      {/* Dams */}
      {dams.map((dam, idx) => (
        <Dam
          key={dam.id}
          position={[(dam.x - 350) / 50, 0, (dam.y - 200) / 50]}
          name={dam.name}
          ecologicalImpact={dam.ecologicalImpact}
          volume={dam.volume}
          maxVolume={dam.maxVolume}
          power={dam.power}
          onClick={() => console.log(`Clicked ${dam.name}`)}
        />
      ))}
      
      {/* Flow particles between dams */}
      {dams.map((dam, idx) => {
        if (idx < dams.length - 1) {
          const nextDam = dams[idx + 1];
          return (
            <FlowParticles
              key={`flow-${idx}`}
              from={[(dam.x - 350) / 50, 0.5, (dam.y - 200) / 50]}
              to={[(nextDam.x - 350) / 50, 0.5, (nextDam.y - 200) / 50]}
              intensity={dam.outflow}
            />
          );
        }
        return null;
      })}
      
      {/* Ecosystem elements */}
      {showEcological && ecosystemElements.map(element => {
        const pos = [(element.x - 350) / 50, 0.2, (element.y - 200) / 50];
        
        if (element.type === 'fish') {
          return (
            <Fish
              key={element.id}
              position={pos}
              health={element.health}
              moving={element.moving}
              blocked={element.blocked}
            />
          );
        }
        
        if (element.type === 'vegetation') {
          return (
            <Vegetation
              key={element.id}
              position={pos}
              health={element.health}
              waterAccess={element.waterAccess}
            />
          );
        }
        
        return null;
      })}
      
      {/* Policy indicator */}
      <Html position={[8, 3, 0]} center>
        <div className="bg-white/90 px-4 py-2 rounded-lg shadow-lg">
          <div className="text-sm font-semibold text-gray-800">
            Active Policy: {selectedPolicy}
          </div>
        </div>
      </Html>
    </>
  );
};

// Main component
const HydroPowerDemo = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedPolicy, setSelectedPolicy] = useState('balanced');
  const [showEcological, setShowEcological] = useState(true);
  
  // Dam system state
  const [dams, setDams] = useState([
    {
      id: 'upstream',
      name: 'Lac Saint-Jean',
      x: 200, y: 150,
      volume: 75, maxVolume: 100, minVolume: 20,
      naturalInflow: 45,
      outflow: 30,
      power: 0,
      ecologicalImpact: 'moderate',
      fishPassage: true,
      sedimentTrap: 60
    },
    {
      id: 'middle',
      name: 'Péribonka Dam', 
      x: 380, y: 220,
      volume: 60, maxVolume: 85, minVolume: 15,
      naturalInflow: 25,
      outflow: 25,
      power: 0,
      ecologicalImpact: 'high',
      fishPassage: false,
      sedimentTrap: 80
    },
    {
      id: 'downstream',
      name: 'Saguenay Outlet',
      x: 520, y: 180,
      volume: 45, maxVolume: 70, minVolume: 10,
      naturalInflow: 20,
      outflow: 20,
      power: 0,
      ecologicalImpact: 'low',
      fishPassage: true,
      sedimentTrap: 30
    }
  ]);

  // Ecosystem elements
  const [ecosystemElements, setEcosystemElements] = useState([
    // Fish populations
    { id: 'fish1', type: 'fish', x: 150, y: 200, health: 0.9, moving: true, blocked: false },
    { id: 'fish2', type: 'fish', x: 250, y: 180, health: 0.7, moving: true, blocked: false },
    { id: 'fish3', type: 'fish', x: 350, y: 240, health: 0.6, moving: false, blocked: true },
    { id: 'fish4', type: 'fish', x: 450, y: 190, health: 0.8, moving: true, blocked: false },
    
    // Vegetation
    { id: 'veg1', type: 'vegetation', x: 120, y: 160, health: 0.9, waterAccess: true },
    { id: 'veg2', type: 'vegetation', x: 180, y: 140, health: 0.8, waterAccess: true },
    { id: 'veg3', type: 'vegetation', x: 320, y: 200, health: 0.4, waterAccess: false },
    { id: 'veg4', type: 'vegetation', x: 480, y: 170, health: 0.7, waterAccess: true }
  ]);

  const [ecosystemHealth, setEcosystemHealth] = useState({
    fishPopulation: 85,
    riparianVegetation: 90,
    waterQuality: 80,
    sedimentFlow: 70,
    floodplainConnectivity: 75
  });

  const policies = {
    economic: {
      name: "Pure Economic Optimization",
      color: "bg-red-600",
      description: "Maximize revenue, ignore ecological impact",
      ecologicalScore: 30,
      strategy: (dam, price) => dam.volume * 0.8
    },
    balanced: {
      name: "Balanced Policy", 
      color: "bg-green-600",
      description: "Balance profit with environmental constraints",
      ecologicalScore: 75,
      strategy: (dam, price, season) => dam.volume * (0.3 + 0.3 * (price / 100))
    },
    ecological: {
      name: "Eco-First Policy",
      color: "bg-blue-600", 
      description: "Prioritize natural flow patterns",
      ecologicalScore: 90,
      strategy: (dam, price, season) => Math.max(10, dam.naturalInflow * 0.8)
    }
  };

  // Animation loop
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentTime(prev => prev + 1);
      
      // Update dam operations based on selected policy
      setDams(prevDams => {
        return prevDams.map(dam => {
          const policy = policies[selectedPolicy];
          const electricityPrice = 60 + Math.sin(currentTime * 0.1) * 30;
          const season = Math.sin(currentTime * 0.05);
          
          const newOutflow = policy.strategy(dam, electricityPrice, season);
          const powerOutput = newOutflow * (dam.volume / dam.maxVolume) * 3;
          
          // Calculate ecological impact
          let impact = 'low';
          if (newOutflow > dam.naturalInflow * 1.5) impact = 'high';
          else if (newOutflow > dam.naturalInflow * 1.2) impact = 'moderate';
          
          return {
            ...dam,
            outflow: newOutflow,
            power: powerOutput,
            ecologicalImpact: impact,
            volume: Math.max(dam.minVolume, 
              Math.min(dam.maxVolume, 
                dam.volume + (dam.naturalInflow - newOutflow) * 0.02 + (Math.random() - 0.5)
              )
            )
          };
        });
      });

      // Update ecosystem health
      setEcosystemHealth(prev => {
        const policy = policies[selectedPolicy];
        const targetHealth = policy.ecologicalScore;
        
        return {
          fishPopulation: prev.fishPopulation + (targetHealth - prev.fishPopulation) * 0.02,
          riparianVegetation: prev.riparianVegetation + (targetHealth - prev.riparianVegetation) * 0.01,
          waterQuality: prev.waterQuality + (targetHealth - prev.waterQuality) * 0.015,
          sedimentFlow: prev.sedimentFlow + (targetHealth - prev.sedimentFlow) * 0.01,
          floodplainConnectivity: prev.floodplainConnectivity + (targetHealth - prev.floodplainConnectivity) * 0.01
        };
      });

      // Update ecosystem elements
      setEcosystemElements(prev => prev.map(element => {
        if (element.type === 'fish') {
          const nearbyDam = dams.find(dam => 
            Math.abs(dam.x - element.x) < 100 && Math.abs(dam.y - element.y) < 50
          );
          
          let blocked = false;
          let healthDelta = 0;
          
          if (nearbyDam) {
            blocked = !nearbyDam.fishPassage && nearbyDam.ecologicalImpact === 'high';
            healthDelta = nearbyDam.ecologicalImpact === 'high' ? -0.002 : 
                         nearbyDam.ecologicalImpact === 'moderate' ? -0.001 : 0.001;
          }
          
          return {
            ...element,
            blocked,
            health: Math.max(0.1, Math.min(1, element.health + healthDelta)),
            moving: !blocked && element.health > 0.3
          };
        }
        
        if (element.type === 'vegetation') {
          const nearbyDam = dams.find(dam => 
            Math.abs(dam.x - element.x) < 80 && Math.abs(dam.y - element.y) < 60
          );
          
          let waterAccess = true;
          let healthDelta = 0;
          
          if (nearbyDam && nearbyDam.volume < nearbyDam.minVolume + 10) {
            waterAccess = false;
            healthDelta = -0.003;
          } else {
            healthDelta = 0.001;
          }
          
          return {
            ...element,
            waterAccess,
            health: Math.max(0.1, Math.min(1, element.health + healthDelta))
          };
        }
        
        return element;
      }));
      
    }, 100);

    return () => clearInterval(interval);
  }, [isPlaying, selectedPolicy, currentTime]);

  const reset = () => {
    setCurrentTime(0);
    setIsPlaying(false);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-green-50 via-blue-50 to-green-100 min-h-screen">
      {/* Header */}
      <div className="text-center mb-6">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">
          3D Watershed Optimization
        </h1>
        <p className="text-gray-600 text-lg max-w-3xl mx-auto">
          Interactive 3D visualization of hydro-power operations and their ecological impact
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white/80 backdrop-blur rounded-2xl p-6 mb-6 shadow-lg border border-green-200">
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center space-x-4">
            <button 
              onClick={() => setIsPlaying(!isPlaying)}
              className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-xl transition-all duration-200 shadow-lg"
            >
              {isPlaying ? <Pause size={24} /> : <Play size={24} />}
              <span className="font-semibold">{isPlaying ? 'Pause' : 'Start'} Simulation</span>
            </button>
            
            <button 
              onClick={reset}
              className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-3 rounded-xl transition-all duration-200"
            >
              <RotateCcw size={20} />
              <span>Reset</span>
            </button>

            <label className="flex items-center space-x-2 text-gray-700">
              <input
                type="checkbox"
                checked={showEcological}
                onChange={(e) => setShowEcological(e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <span>Show Ecosystem</span>
            </label>
          </div>

          <div className="text-right">
            <div className="text-sm text-gray-600">Simulation Time</div>
            <div className="text-2xl font-bold text-gray-800">{Math.floor(currentTime / 10)} days</div>
          </div>
        </div>

        {/* Policy Selection */}
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(policies).map(([key, policy]) => (
            <button
              key={key}
              onClick={() => setSelectedPolicy(key)}
              className={`p-4 rounded-xl border-2 transition-all duration-300 ${
                selectedPolicy === key 
                  ? `${policy.color} text-white border-white shadow-lg scale-105` 
                  : 'bg-white border-gray-300 text-gray-700 hover:border-gray-400 hover:shadow-md'
              }`}
            >
              <div className="font-bold text-lg">{policy.name}</div>
              <div className="text-sm opacity-90 mt-1">{policy.description}</div>
              <div className="text-xs mt-2 font-medium">
                Ecological Score: {policy.ecologicalScore}/100
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 3D Scene */}
      <div className="bg-white/60 backdrop-blur rounded-2xl shadow-xl border border-blue-200 h-96 mb-6">
        <Canvas
          camera={{
            type: 'OrthographicCamera',
            position: [10, 10, 10],
            zoom: 50,
            near: 0.1,
            far: 1000
          }}
        >
          <OrbitControls 
            enablePan={true} 
            enableZoom={true} 
            enableRotate={true}
            maxPolarAngle={Math.PI / 2}
            minPolarAngle={0}
          />
          
          <HydroScene 
            dams={dams}
            ecosystemElements={ecosystemElements}
            selectedPolicy={selectedPolicy}
            showEcological={showEcological}
          />
        </Canvas>
      </div>

      {/* Ecosystem Health Dashboard */}
      {showEcological && (
        <div className="bg-white/80 backdrop-blur rounded-2xl p-6 shadow-lg border border-green-200">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 flex items-center">
            <Leaf className="mr-3 text-green-600" size={28} />
            Ecosystem Health Indicators
          </h2>
          
          <div className="grid grid-cols-5 gap-6">
            {Object.entries(ecosystemHealth).map(([key, value]) => {
              const getColor = (val) => val > 70 ? 'text-green-600' : val > 40 ? 'text-yellow-600' : 'text-red-600';
              const getBgColor = (val) => val > 70 ? 'bg-green-100' : val > 40 ? 'bg-yellow-100' : 'bg-red-100';
              
              const icons = {
                fishPopulation: <FishIcon size={24} />,
                riparianVegetation: <div className="w-6 h-6 bg-green-600 rounded-full" />,
                waterQuality: <div className="w-6 h-6 bg-blue-600 rounded" />,
                sedimentFlow: <div className="w-6 h-6 rounded-full bg-amber-600" />,
                floodplainConnectivity: <div className="w-6 h-6 border-2 border-blue-600 rounded" />
              };
              
              return (
                <div key={key} className={`p-4 rounded-xl ${getBgColor(value)} border border-gray-200`}>
                  <div className={`flex items-center justify-center mb-2 ${getColor(value)}`}>
                    {icons[key]}
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-600 capitalize mb-1">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </div>
                    <div className={`text-2xl font-bold ${getColor(value)}`}>
                      {Math.round(value)}%
                    </div>
                  </div>
                  
                  {/* Health bar */}
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-1000 ${
                        value > 70 ? 'bg-green-500' : value > 40 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${value}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default HydroPowerDemo;