#!/bin/bash
# Download graphs from SuiteSparse Matrix Collection
# Usage: ./scripts/download_graph.sh GROUP NAME

if [ $# -lt 2 ]; then
    echo "Usage: $0 GROUP NAME"
    echo ""
    echo "Examples:"
    echo "  $0 SNAP roadNet-CA"
    echo "  $0 DIMACS10 europe_osm"
    echo "  $0 SNAP com-Orkut"
    echo ""
    echo "Popular choices for MPI:"
    echo "  SNAP/roadNet-CA       - 1.9M vertices, road network (good locality)"
    echo "  SNAP/roadNet-PA       - 1.1M vertices, road network"
    echo "  SNAP/roadNet-TX       - 1.4M vertices, road network"
    echo "  DIMACS10/europe_osm   - 50M vertices, road network (LARGE!)"
    echo "  vanHeukelum/cage15    - 5.2M vertices, 3D mesh"
    echo "  SNAP/com-Orkut        - 3.1M vertices, social network"
    exit 1
fi

GROUP=$1
NAME=$2
URL="https://suitesparse-collection-website.herokuapp.com/MM/${GROUP}/${NAME}.tar.gz"

echo "Downloading ${NAME} from ${GROUP}..."
echo "URL: ${URL}"

# Create data directory
mkdir -p data
cd data

# Download
echo "Downloading..."
wget -q --show-progress "${URL}" || {
    echo "Error: Download failed. Check that the group and name are correct."
    exit 1
}

# Extract
echo "Extracting..."
tar -xzf "${NAME}.tar.gz"

# Move to data directory
if [ -f "${GROUP}/${NAME}/${NAME}.mtx" ]; then
    mv "${GROUP}/${NAME}/${NAME}.mtx" ./
    echo "Success! Graph saved to: data/${NAME}.mtx"

    # Show graph info
    echo ""
    echo "Graph info:"
    head -20 "${NAME}.mtx" | grep -E "^%|^[0-9]" | head -5

    # Cleanup
    rm -rf "${GROUP}" "${NAME}.tar.gz"
else
    echo "Error: Could not find ${NAME}.mtx in archive"
    ls -R "${GROUP}/"
    exit 1
fi

echo ""
echo "To test with MPI:"
echo "  mpirun -np 2 ./build/cc_mpi data/${NAME}.mtx 0"
