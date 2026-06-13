#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <optional>
#include <iomanip>
#include <chrono>
#include <string>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

// ── Constants ─────────────────────────────────────────────────────────────────
constexpr double TWO_PI      = 2.0 * M_PI;
constexpr double GROUND_Z    = -0.5;
constexpr double OMEGA       = 8.0;      // rad/s — actuation wave speed
constexpr double DT          = 0.005;    // s — stable below 2/ω_natural
constexpr double TOTAL_TIME  = 3.0;      // s — evaluation window
constexpr double SPRING_K    = 3000.0;   // N/m
constexpr double SPRING_DAMP = 30.0;     // Ns/m — dashpot
constexpr double NODE_MASS   = 0.1;      // kg
constexpr double GRAVITY     = -9.81;    // m/s²
constexpr double RESTITUTION = 0.4;
constexpr double FRICTION    = 0.7;      // Coulomb coefficient
constexpr double AMP_MIN     = 0.0;
constexpr double AMP_MAX     = 0.5;

// ── Vec3 ──────────────────────────────────────────────────────────────────────
struct Vec3 {
    double x, y, z;
    Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    Vec3 operator*(double s)      const { return {x*s,   y*s,   z*s};   }
    Vec3 operator/(double s)      const { return {x/s,   y/s,   z/s};   }
    Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    Vec3& operator-=(const Vec3& v) { x-=v.x; y-=v.y; z-=v.z; return *this; }
    double dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    double norm() const { return std::sqrt(x*x + y*y + z*z); }
};

struct Node {
    Vec3   pos;
    Vec3   vel   = {0, 0, 0};
    Vec3   force = {0, 0, 0};
    double mass  = NODE_MASS;
};

struct Spring {
    int    n1_idx, n2_idx;
    double k       = SPRING_K;
    double damping = SPRING_DAMP;
    double L0;
    bool   is_actuator = false;
    double amplitude   = 0.0;
    double phase       = 0.0;
};

// ── CPG Genome ────────────────────────────────────────────────────────────────
// 4 parameters encode bilateral symmetry + diagonal gait coupling.
// Reduces search space from 8 → 4 and guarantees straight locomotion.
struct Genome {
    double arm_amp;    // arm muscle amplitude   [0, AMP_MAX]
    double leg_amp;    // leg muscle amplitude   [0, AMP_MAX]
    double base_phase; // phase of left arm      [0, 2π]
    double diag_off;   // leg phase offset       [0, 2π]
};

static inline double wrap(double p) {
    p = std::fmod(p, TWO_PI);
    return p < 0.0 ? p + TWO_PI : p;
}

// ── Skeleton ──────────────────────────────────────────────────────────────────
void create_voxelbot(std::vector<Node>& nodes, std::vector<Spring>& springs) {
    nodes.push_back({{ 1.0,  0.5,  0.8}}); // 0 Left Shoulder
    nodes.push_back({{ 1.0, -0.5,  0.8}}); // 1 Right Shoulder
    nodes.push_back({{-1.0,  0.4,  0.3}}); // 2 Left Hip
    nodes.push_back({{-1.0, -0.4,  0.3}}); // 3 Right Hip
    nodes.push_back({{ 1.5,  0.7, -0.5}}); // 4 Left Hand  (ground level)
    nodes.push_back({{ 1.5, -0.7, -0.5}}); // 5 Right Hand (ground level)
    nodes.push_back({{-1.5,  0.5, -0.5}}); // 6 Left Foot  (ground level)
    nodes.push_back({{-1.5, -0.5, -0.5}}); // 7 Right Foot (ground level)

    auto add = [&](int a, int b, bool muscle) {
        Spring s;
        s.n1_idx = a; s.n2_idx = b;
        s.L0 = (nodes[a].pos - nodes[b].pos).norm();
        s.k = SPRING_K; s.damping = SPRING_DAMP;
        s.is_actuator = muscle;
        springs.push_back(s);
    };

    // Rigid torso (passive)
    add(0,1,false); add(2,3,false);
    add(0,2,false); add(1,3,false);
    add(0,3,false); add(1,2,false);
    // Front limbs — actuators at indices [6, 7]
    add(4,0,true);  add(5,1,true);
    add(4,2,false); add(5,3,false);
    // Back limbs — actuators at indices [10, 11]
    add(6,2,true);  add(7,3,true);
    add(6,0,false); add(7,1,false);
    // Lateral stabilizers
    add(4,1,false); add(5,0,false);
    add(6,3,false); add(7,2,false);
}

// Actuator order from create_voxelbot: [arm_L, arm_R, leg_L, leg_R]
void apply_genome(const Genome& g, std::vector<Spring>& springs) {
    int act[4], ai = 0;
    for (int i = 0; i < (int)springs.size(); ++i)
        if (springs[i].is_actuator) act[ai++] = i;

    springs[act[0]].amplitude = g.arm_amp;
    springs[act[0]].phase     = g.base_phase;
    springs[act[1]].amplitude = g.arm_amp;
    springs[act[1]].phase     = wrap(g.base_phase + M_PI);
    springs[act[2]].amplitude = g.leg_amp;
    springs[act[2]].phase     = wrap(g.base_phase + g.diag_off);
    springs[act[3]].amplitude = g.leg_amp;
    springs[act[3]].phase     = wrap(g.base_phase + g.diag_off + M_PI);
}

// ── Physics ───────────────────────────────────────────────────────────────────
void simulate_step(std::vector<Node>& nodes, const std::vector<Spring>& springs,
                   double t, double dt) {
    for (auto& n : nodes)
        n.force = {0.0, 0.0, n.mass * GRAVITY};

    for (const auto& sp : springs) {
        Vec3   delta = nodes[sp.n2_idx].pos - nodes[sp.n1_idx].pos;
        double L     = delta.norm();
        if (L < 1e-9) continue;
        Vec3 dir = delta / L;

        double target_L = sp.L0;
        if (sp.is_actuator)
            target_L += sp.amplitude * std::sin(OMEGA * t + sp.phase);

        double spring_f = sp.k * (L - target_L);
        double damp_f   = sp.damping * (nodes[sp.n2_idx].vel - nodes[sp.n1_idx].vel).dot(dir);

        Vec3 f = dir * (spring_f + damp_f);
        nodes[sp.n1_idx].force += f;
        nodes[sp.n2_idx].force -= f;
    }

    for (auto& n : nodes) {
        n.vel += (n.force / n.mass) * dt;
        n.pos += n.vel * dt;

        if (n.pos.z < GROUND_Z) {
            n.pos.z = GROUND_Z;
            if (n.vel.z < 0.0) n.vel.z *= -RESTITUTION;
            double normal_f    = std::max(0.0, -n.force.z);
            double friction_dv = FRICTION * normal_f * dt / n.mass;
            double tan_speed   = std::hypot(n.vel.x, n.vel.y);
            if (tan_speed > 1e-9) {
                double scale = std::max(0.0, tan_speed - friction_dv) / tan_speed;
                n.vel.x *= scale;
                n.vel.y *= scale;
            }
        }
    }
}

// ── Fitness ───────────────────────────────────────────────────────────────────
double evaluate_fitness(const Genome& g) {
    std::vector<Node>   nodes;
    std::vector<Spring> springs;
    create_voxelbot(nodes, springs);
    apply_genome(g, springs);

    const double n_inv = 1.0 / nodes.size();
    double init_x = 0.0;
    for (const auto& n : nodes) init_x += n.pos.x;
    init_x *= n_inv;

    for (double t = 0.0; t < TOTAL_TIME; t += DT) {
        simulate_step(nodes, springs, t, DT);
        for (const auto& n : nodes)
            if (std::abs(n.pos.x) > 50.0 || std::abs(n.pos.z) > 50.0)
                return -500.0; // explosion guard
    }

    double final_x = 0.0, final_y = 0.0;
    for (const auto& n : nodes) { final_x += n.pos.x; final_y += n.pos.y; }
    final_x *= n_inv; final_y *= n_inv;

    return (final_x - init_x) - std::abs(final_y);
}

// ── GA primitives ─────────────────────────────────────────────────────────────
struct Robot {
    Genome                genome;
    std::optional<double> fitness; // empty = unevaluated
};

int tournament(int pool, int k, std::mt19937& rng, const std::vector<Robot>& pop) {
    std::uniform_int_distribution<int> d(0, pool - 1);
    int best = d(rng);
    for (int i = 1; i < k; ++i) {
        int c = d(rng);
        if (pop[c].fitness.value() > pop[best].fitness.value()) best = c;
    }
    return best;
}

Genome random_genome(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    return { u(rng)*AMP_MAX, u(rng)*AMP_MAX, u(rng)*TWO_PI, u(rng)*TWO_PI };
}

Genome crossover(const Genome& a, const Genome& b, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    return {
        u(rng) > 0.5 ? a.arm_amp    : b.arm_amp,
        u(rng) > 0.5 ? a.leg_amp    : b.leg_amp,
        u(rng) > 0.5 ? a.base_phase : b.base_phase,
        u(rng) > 0.5 ? a.diag_off   : b.diag_off,
    };
}

void mutate(Genome& g, double std_dev, std::mt19937& rng) {
    constexpr double MUT_RATE = 0.25; // per-parameter (expected ~1 mutation per child)
    std::normal_distribution<double> n(0.0, std_dev);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    if (u(rng)<MUT_RATE) { g.arm_amp    += n(rng);         g.arm_amp    = std::clamp(g.arm_amp,    AMP_MIN, AMP_MAX); }
    if (u(rng)<MUT_RATE) { g.leg_amp    += n(rng);         g.leg_amp    = std::clamp(g.leg_amp,    AMP_MIN, AMP_MAX); }
    if (u(rng)<MUT_RATE) { g.base_phase += n(rng)*TWO_PI;  g.base_phase = wrap(g.base_phase); }
    if (u(rng)<MUT_RATE) { g.diag_off   += n(rng)*TWO_PI;  g.diag_off   = wrap(g.diag_off); }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    auto t0 = std::chrono::steady_clock::now();
    std::cout << "=== Chimpanzee Voxelbot — CPG Island GA ===\n\n";

    // ── Hyperparameters ────────────────────────────────────────────────────
    constexpr int    N_ISLANDS    = 4;
    constexpr int    ISLAND_SIZE  = 25;   // total population = 100
    constexpr int    GENS         = 2000;
    constexpr int    ISLAND_ELITE = 2;    // 8% elitism per island
    constexpr int    TOURN_K      = 3;
    constexpr double MUT_STD_I    = 0.15; // initial (wide exploration)
    constexpr double MUT_STD_F    = 0.02; // final   (fine-tuning)
    constexpr int    MIGRATE_EVERY= 200;  // ring migration interval
    constexpr int    STAG_LIMIT   = 300;  // gens before diversity injection
    constexpr int    CKPT_EVERY   = 200;

    // Per-island RNGs — required for thread-safe evolution in step G
    std::random_device rd;
    std::array<std::mt19937, N_ISLANDS> rngs;
    for (auto& r : rngs) r = std::mt19937(rd());

    // ── Initialize islands ─────────────────────────────────────────────────
    std::array<std::vector<Robot>, N_ISLANDS> islands;
    for (int isl = 0; isl < N_ISLANDS; ++isl) {
        islands[isl].resize(ISLAND_SIZE);
        for (auto& rob : islands[isl])
            rob.genome = random_genome(rngs[isl]);
    }

    std::cout << "Islands=" << N_ISLANDS << "  Size=" << ISLAND_SIZE
              << "  Total=" << N_ISLANDS*ISLAND_SIZE
              << "  Gens=" << GENS << "  MigrateEvery=" << MIGRATE_EVERY << "\n\n";

    // ── Fitness log ────────────────────────────────────────────────────────
    std::ofstream log("fitness_log.csv");
    if (!log.is_open()) { std::cerr << "Cannot open fitness_log.csv\n"; return 1; }
    log << "Generation,GlobalBest,I0Best,I1Best,I2Best,I3Best,GlobalMean,GlobalStdDev\n";

    double alltime_best = -1e9;
    int    stag_count   = 0;

    // ── Evolutionary loop ──────────────────────────────────────────────────
    for (int g = 0; g < GENS; ++g) {

        // Anneal mutation std: wide exploration → fine-tuning
        double alpha   = static_cast<double>(g) / std::max(1, GENS - 1);
        double mut_std = MUT_STD_I + alpha * (MUT_STD_F - MUT_STD_I);

        // A. Parallel fitness evaluation (all 100 robots across all islands)
        #pragma omp parallel for schedule(dynamic)
        for (int flat = 0; flat < N_ISLANDS * ISLAND_SIZE; ++flat) {
            int isl = flat / ISLAND_SIZE;
            int rob = flat % ISLAND_SIZE;
            auto& r = islands[isl][rob];
            if (!r.fitness.has_value())
                r.fitness = evaluate_fitness(r.genome);
        }

        // B. Per-island sort (descending fitness)
        for (int isl = 0; isl < N_ISLANDS; ++isl) {
            std::sort(islands[isl].begin(), islands[isl].end(),
                [](const Robot& a, const Robot& b) {
                    return a.fitness.value() > b.fitness.value();
                });
        }

        // C. Stats
        double global_best = -1e9, sum = 0.0, sum2 = 0.0;
        std::array<double, N_ISLANDS> ibest;
        for (int isl = 0; isl < N_ISLANDS; ++isl) {
            ibest[isl] = islands[isl][0].fitness.value();
            global_best = std::max(global_best, ibest[isl]);
            for (const auto& r : islands[isl]) {
                double f = r.fitness.value();
                sum += f; sum2 += f * f;
            }
        }
        double total  = N_ISLANDS * ISLAND_SIZE;
        double mean   = sum / total;
        double stddev = std::sqrt(std::max(0.0, sum2 / total - mean * mean));

        log << g << "," << global_best
            << "," << ibest[0] << "," << ibest[1] << "," << ibest[2] << "," << ibest[3]
            << "," << mean << "," << stddev << "\n";

        if (g % 10 == 0 || g == GENS - 1) {
            std::cout << "Gen " << std::setw(4) << g
                      << "  Best=" << std::setw(7) << std::fixed << std::setprecision(4) << global_best
                      << "  Mean=" << std::setw(7) << mean
                      << "  σ="    << std::setw(6) << stddev
                      << "  Islands:[";
            for (int isl = 0; isl < N_ISLANDS; ++isl)
                std::cout << std::setw(6) << ibest[isl] << (isl < N_ISLANDS-1 ? " " : "");
            std::cout << "]\n";
        }

        // D. Stagnation → reinitialise bottom half of each island
        if (global_best > alltime_best + 1e-6) { alltime_best = global_best; stag_count = 0; }
        else ++stag_count;

        if (stag_count >= STAG_LIMIT) {
            std::cout << "  [Stagnation at gen " << g << "] Reinitialising bottom half\n";
            for (int isl = 0; isl < N_ISLANDS; ++isl) {
                for (int i = ISLAND_SIZE / 2; i < ISLAND_SIZE; ++i) {
                    islands[isl][i].genome = random_genome(rngs[isl]);
                    islands[isl][i].fitness.reset();
                }
            }
            stag_count = 0;
        }

        // E. Checkpoint
        if (g % CKPT_EVERY == 0) {
            // Find global best robot
            int best_isl = 0;
            for (int isl = 1; isl < N_ISLANDS; ++isl)
                if (ibest[isl] > ibest[best_isl]) best_isl = isl;

            std::vector<Node> tn; std::vector<Spring> ts;
            create_voxelbot(tn, ts);
            apply_genome(islands[best_isl][0].genome, ts);

            std::ofstream ck("checkpoint_gen" + std::to_string(g) + ".csv");
            ck << "SpringIndex,IsActuator,Amplitude,Phase\n";
            for (size_t i = 0; i < ts.size(); ++i)
                ck << i << "," << ts[i].is_actuator << "," << ts[i].amplitude << "," << ts[i].phase << "\n";
        }

        if (g == GENS - 1) break;

        // F. Ring migration (0→1→2→3→0, replace worst of destination)
        if (g > 0 && g % MIGRATE_EVERY == 0) {
            std::cout << "  [Migration gen=" << g << "]\n";
            std::array<Robot, N_ISLANDS> migrants;
            for (int isl = 0; isl < N_ISLANDS; ++isl)
                migrants[isl] = islands[isl][0];
            for (int isl = 0; isl < N_ISLANDS; ++isl) {
                int dst = (isl + 1) % N_ISLANDS;
                islands[dst].back() = migrants[isl];
                islands[dst].back().fitness.reset();
            }
        }

        // G. Build next generation per island (sequential — uses per-island RNGs)
        for (int isl = 0; isl < N_ISLANDS; ++isl) {
            auto& pop = islands[isl];
            auto& rng = rngs[isl];
            int   parent_pool = ISLAND_SIZE / 2;

            std::vector<Robot> next;
            next.reserve(ISLAND_SIZE);

            // Elitism
            for (int i = 0; i < ISLAND_ELITE; ++i)
                next.push_back(pop[i]);

            // Tournament + crossover + mutation
            while ((int)next.size() < ISLAND_SIZE) {
                int p1 = tournament(parent_pool, TOURN_K, rng, pop);
                int p2; do { p2 = tournament(parent_pool, TOURN_K, rng, pop); } while (p2 == p1);

                Robot child;
                child.genome = crossover(pop[p1].genome, pop[p2].genome, rng);
                mutate(child.genome, mut_std, rng);
                next.push_back(std::move(child));
            }
            pop = std::move(next);
        }
    }

    log.close();

    // ── Export global winner ───────────────────────────────────────────────
    int    best_isl = 0;
    double best_f   = islands[0][0].fitness.value();
    for (int isl = 1; isl < N_ISLANDS; ++isl)
        if (islands[isl][0].fitness.value() > best_f) { best_f = islands[isl][0].fitness.value(); best_isl = isl; }

    const Genome& W = islands[best_isl][0].genome;
    std::cout << "\nEvolution complete.\n"
              << "  All-time best: " << alltime_best << "\n"
              << "  Final best:    " << best_f << "\n"
              << "  Genome: arm_amp=" << W.arm_amp
              << "  leg_amp=" << W.leg_amp
              << "  base_phase=" << W.base_phase
              << "  diag_off=" << W.diag_off << "\n";

    // Derive spring parameters from genome for the visualiser
    {
        std::vector<Node> tn; std::vector<Spring> ts;
        create_voxelbot(tn, ts);
        apply_genome(W, ts);

        std::ofstream out("best_parameters.csv");
        if (!out.is_open()) { std::cerr << "Cannot open best_parameters.csv\n"; return 1; }
        out << "SpringIndex,IsActuator,Amplitude,Phase\n";
        for (size_t i = 0; i < ts.size(); ++i)
            out << i << "," << ts[i].is_actuator << "," << ts[i].amplitude << "," << ts[i].phase << "\n";
    }

    // Human-readable genome export
    {
        std::ofstream out("genome_best.csv");
        out << "ArmAmplitude,LegAmplitude,BasePhase,DiagonalOffset\n";
        out << W.arm_amp << "," << W.leg_amp << "," << W.base_phase << "," << W.diag_off << "\n";
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "Wall time: " << std::fixed << std::setprecision(1) << elapsed << " s\n";
    std::cout << "Outputs: best_parameters.csv  genome_best.csv  fitness_log.csv  checkpoint_gen*.csv\n";
    std::cout << "Compile: g++ -O3 -march=native -fopenmp -std=c++17 PE_M_GA.cpp -o chimp_ga\n";
    return 0;
}
