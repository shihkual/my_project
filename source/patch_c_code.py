code_patch_SQWELL = """

// initialize the patches
const vec3<float> verts[] = {{
    {patch_locations}
    }};

// initialize pair energy
float pair_eng = 0.0;

// set the square well parameters and number of patches
const float epsilon = {epsilon:.15f};
const float repulsive_epsilon = {repulsive_epsilon:.15f};
const float sigma = {sigma:.15f};
const float lambdasigma = {lambdasigma:.15f};
const float repulsive_radius = {repulsive_radius:.15f};
const int n_patches = {n_patches};

// first check for hard core repulsion
float rsq = dot(r_ij, r_ij);
if (rsq < 4 * repulsive_radius * repulsive_radius)
    {{
      pair_eng += repulsive_epsilon;
    }}

// check patch overlaps
for (int patch_idx_on_i = 0; patch_idx_on_i < n_patches; ++patch_idx_on_i)
    {{
      // let r_m be the location of the patch on particle i
      vec3<float> r_m = rotate(q_i, verts[patch_idx_on_i]);

      // for each of these patches on particle i, loop through patches on j
      for (int patch_idx_on_j = 0; patch_idx_on_j < n_patches; ++patch_idx_on_j)
      {{
        // let r_n be the location of the patch on particle j
        vec3<float> r_n = rotate(q_j, verts[patch_idx_on_j]) + r_ij;

        // now the vector from r_m to r_n is just r_n - r_m
        // call that vector dr
        vec3<float> dr = r_n - r_m;

        // now check to see if the length of dr is less than the range of the square well
        float rsq = dot(dr, dr);
        if (rsq <= lambdasigma*lambdasigma)  // assumes sigma = 0
        {{
        pair_eng += -epsilon;
        }}
      }}
    }}

return pair_eng;
"""

code_patch_KF_triangle = """
// initialize the patches
const vec3<float> verts[] = {{
    {patch_locations}
    }};
const vec3<float> patch_dir_1[] = {{
    {patch_directions_1}
    }};
const vec3<float> patch_dir_2[] = {{
    {patch_directions_2}
    }};

// initialize pair energy
float pair_eng = 0.0;
float orientational_component = 1.0;

// set the square well parameters and number of patches
const float epsilon = {epsilon:.15f};
const float repulsive_epsilon = {repulsive_epsilon:.15f};
const float lambdasigma = {lambdasigma:.15f};
const float repulsive_radius = {repulsive_radius:.15f};
const float cos_half_alpha = {patch_angle:.15f};
const int n_patches = {n_patches};
const bool check_same_or_guest = type_i == 2 || type_j == 2 || type_i == type_j; 

// first check for hard core repulsion
float rsq = dot(r_ij, r_ij);
if (check_same_or_guest) 
    {{
    ;
    }} 
else 
    {{
        if (rsq < 4 * repulsive_radius * repulsive_radius) 
            {{
            pair_eng += repulsive_epsilon;
            }}
    }}
// check patch overlaps
bool patch_m_aligned = false;
bool patch_n_aligned = false;
if (check_same_or_guest) 
    {{
    ;
    }} 
else 
    {{
    for (int patch_idx_on_i = 0; patch_idx_on_i < n_patches; ++patch_idx_on_i)
        {{
          // for particle i
          // let r_m be the location of the patches
          // patch_dir_m be the unit vector of direcion of patches
          vec3<float> r_m = rotate(q_i, verts[patch_idx_on_i]);

          // for each of these patches on particle i, loop through patches on j
          for (int patch_idx_on_j = 0; patch_idx_on_j < n_patches; ++patch_idx_on_j)
              {{
                // for particle j
                // let r_n be the location of the patches
                // patch_dir_n be the unit vector of direcion of patches
                vec3<float> r_n = rotate(q_j, verts[patch_idx_on_j]) + r_ij;

                // now the vector from r_m to r_n is just r_n - r_m
                // call that vector dr
                vec3<float> dr = r_n - r_m;
                float dr_sq = dot(dr, dr);
                vec3<float> dr_hat = dr / sqrt(dr_sq);

                // check if patches on particles i and j are aligned with dr vector
                if (type_i == 0)
                    {{
                    patch_m_aligned = dot(rotate(q_i, patch_dir_1[patch_idx_on_i]), dr_hat) >= cos_half_alpha;
                    patch_n_aligned = dot(rotate(q_j, patch_dir_2[patch_idx_on_j]), -dr_hat) >= cos_half_alpha;
                    }}
                else
                    {{
                    patch_m_aligned = dot(rotate(q_i, patch_dir_2[patch_idx_on_i]), dr_hat) >= cos_half_alpha;
                    patch_n_aligned = dot(rotate(q_j, patch_dir_1[patch_idx_on_j]), -dr_hat) >= cos_half_alpha;
                    }}

                // check angular part of K-F potential
                if (patch_m_aligned && patch_n_aligned)
                    {{
                    orientational_component = 1.0;
                    }}
                else
                    {{
                    orientational_component = 0.0;
                    }}


                // now check to see if the length of dr is less than the range of the square well        
                 if (dr_sq <= lambdasigma*lambdasigma)  
                    {{
                    pair_eng += -epsilon * orientational_component;
                    }}
             }}
        }}
    }}
return pair_eng;
"""
