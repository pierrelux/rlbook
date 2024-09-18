// water_heater_viz.js

console.log("water_heater_viz.js loaded");

function initializeVisualization() {
    console.log("Initializing visualization");

    const width = 300;
    const height = 400;
    const svg = d3.select("#water-heater")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    // Add a simple rectangle to the SVG
    svg.append("rect")
        .attr("x", 50)
        .attr("y", 50)
        .attr("width", 200)
        .attr("height", 300)
        .attr("fill", "blue");

    console.log("SVG created");
}

// Call the initialization function when the page loads
window.onload = function() {
    console.log("Window loaded");
    initializeVisualization();
};