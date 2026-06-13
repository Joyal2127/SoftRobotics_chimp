#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>

// ... [Keep the previous Vec3, Node, Spring structs, simulate_step, and evaluate_fitness functions] ...

// Wrapper to hold a robot's DNA and its score
struct Robot {
    std::vector<Spring> springs;
    double fitness = -1000.0;
};

int main() {
    std::cout << "Initializing GA for Chimpanzee Voxelbot..." << std::endl;
    
    int POPULATION_SIZE = 100;
    int GENERATIONS = 2000;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.1); 

    // 1. Initialize Population Matrix
    std::vector<Robot> population(POPULATION_SIZE);
    std::vector<Node> init_nodes; // Dummy nodes to generate the skeleton
    
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        init_nodes.clear();
        create_voxelbot(init_nodes, population[i].springs);
        
        // Randomize the starting DNA for generation 0
        for (auto& spring : population[i].springs) {
            if (spring.is_actuator) {
                spring.amplitude = unif(gen) * 0.4; // Random between 0.0 and 0.4
                spring.phase = unif(gen) * 6.28;    // Random between 0 and 2*PI
            }
        }
    }

    std::cout << "Starting Evolution: " << POPULATION_SIZE << " robots for " << GENERATIONS << " generations.\n";

    // 2. The Evolutionary Loop
    for (int gen_idx = 0; gen_idx < GENERATIONS; ++gen_idx) {
        
        // A. Evaluate Fitness
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            // Only re-evaluate if we haven't already (saves massive compute time for elites)
            if (population[i].fitness == -1000.0) { 
                population[i].fitness = evaluate_fitness(population[i].springs);
            }
        }

        // B. Sort Population by Fitness (Descending)
        std::sort(population.begin(), population.end(), [](const Robot& a, const Robot& b) {
            return a.fitness > b.fitness;
        });

        // Print progress
        if (gen_idx % 10 == 0 || gen_idx == GENERATIONS - 1) {
            std::cout << "Generation " << gen_idx << " | Best Distance: " << population[0].fitness << "\n";
        }

        // C. Culling & Crossover (Keep top 50%, replace bottom 50%)
        int half_pop = POPULATION_SIZE / 2;
        std::uniform_int_distribution<int> parent_dist(0, half_pop - 1);

        for (int i = half_pop; i < POPULATION_SIZE; ++i) {
            // Select two random parents from the elite top 50%
            int p1_idx = parent_dist(gen);
            int p2_idx = parent_dist(gen);
            
            Robot child;
            child.springs = population[p1_idx].springs; // Copy structure from Parent 1

            // Whole-Muscle Crossover
            for (size_t s = 0; s < child.springs.size(); ++s) {
                if (child.springs[s].is_actuator) {
                    if (unif(gen) > 0.5) {
                        // Inherit this muscle from Parent 2 instead
                        child.springs[s].amplitude = population[p2_idx].springs[s].amplitude;
                        child.springs[s].phase = population[p2_idx].springs[s].phase;
                    }
                    
                    // D. Mutation (10% chance to mutate a muscle)
                    if (unif(gen) < 0.10) {
                        child.springs[s].amplitude += noise(gen);
                        child.springs[s].phase += noise(gen);
                        
                        // Clamp values to physical limits
                        if (child.springs[s].amplitude < 0.0) child.springs[s].amplitude = 0.0;
                        if (child.springs[s].amplitude > 0.5) child.springs[s].amplitude = 0.5;
                    }
                }
            }
            
            // Replace the dead robot with the new child
            population[i] = child;
        }
    }

    // 3. Export the Global Winner
    std::cout << "\nEvolution complete. Exporting global optimum..." << std::endl;
    std::ofstream outfile("best_parameters.csv");
    outfile << "SpringIndex,Amplitude,Phase\n";
    for (size_t i = 0; i < population[0].springs.size(); ++i) {
        outfile << i << "," << population[0].springs[i].amplitude << "," << population[0].springs[i].phase << "\n";
    }
    outfile.close();
    
    return 0;
}