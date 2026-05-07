# Memory & Data 



## Memory summary (1M, before omptimization), problems - duplicated masks, huge size of non-sparse binary matrices
## Projected sizes of objects for 32M doesnt fit in the resources constraints

| Name                          | Size (MB)|    Shape     |   Dtype  	|
|-------------------------------|----------|--------------|--------		|
| movies                        | 0.6      | —            | —      		|
| catalog                       | 0.9      | —            | —     		|
| hybrid.content.user_profiles  | 0.8      | (5954, 18)   | float64 	|
| hybrid.content.user_masks     | 18.4     | (5954, 3233) | bool   		|
| hybrid.collab.U               | 2.3      | (5954, 50)   | float64 	|
| hybrid.collab.Vh              | 1.2      | (50, 3233)   | float64 	|
| hybrid.collab.user_masks      | 18.4     | (5954, 3233) | bool   		|
| popularity.scores             | 0        | (3233,)      | float64 	|

| Metric                                   | Value   	|
|------------------------------------------|---------	|
| Total (sum of tracked objects)           | 42.6 MB 	|
| Peak traced memory (Python only)         | 44.26 MB 	|
| RSS increase (real process memory)       | 57.1 MB 	|

## 32M after dedup of rated indices
_Run: 2026-04-26 12:59:47_

| Component                          | Size      | Shape          | Dtype   |
|------------------------------------|-----------|----------------|---------|
| Python+deps baseline               |   142.1 MB | —              | —       |
| Traced memory (current)            |  291.77 MB | —              | —       |
| Traced memory (peak)               |  561.52 MB | —              | —       |
| Post-load RSS                      |  1018.0 MB | —              | —       |
| RSS diff                           |   875.9 MB | —              | —       |
| movies                             |     5.9 MB | —              | —       |
| catalog                            |    14.1 MB | —              | —       |
| hybrid.content.user_profiles       |    29.8 MB | (195544, 20)   | float64 |
| hybrid.content.rated_indices_      |   146.7 MB | —              | —       |
| hybrid.collab.U                    |    74.6 MB | (195544, 50)   | float64 |
| hybrid.collab.Vh                   |    11.6 MB | (50, 30521)    | float64 |
| hybrid.collab.rated_indices_       |   146.7 MB | —              | —       |
| popularity.scores                  |     0.2 MB | (30521,)       | float64 |


## 32M after re-designing rated indices to avoid arrays overhead
_Run: 2026-04-26 15:24:43_

| Component                          | Size      | Shape          | Dtype   |
|------------------------------------|-----------|----------------|---------|
| Python+deps baseline               |   142.3 MB | —              | —       |
| Traced memory (current)            |  269.31 MB | —              | —       |
| Traced memory (peak)               |  394.84 MB | —              | —       |
| Post-load RSS                      |   484.5 MB | —              | —       |
| RSS diff                           |   342.1 MB | —              | —     |
| movies                             |     5.9 MB | —              | —       |
| catalog                            |    14.1 MB | —              | —       |
| hybrid.content.user_profiles       |    29.8 MB | (195544, 20)   | float64 |
| hybrid.content.rated_indices_      |   121.0 MB | (31720336,)    | int32   |
| hybrid.collab.U                    |    74.6 MB | (195544, 50)   | float64 |
| hybrid.collab.Vh                   |    11.6 MB | (50, 30521)    | float64 |
| hybrid.collab.rated_indices_       |   121.0 MB | (31720336,)    | int32   |
| popularity.scores                  |     0.2 MB | (30521,)       | float64 |


## polars-optimizations for i/o speed and removing pandas from deps
_Run: 2026-04-27 15:38:54_

| Component                          | Size      | Shape          | Dtype   |
|------------------------------------|-----------|----------------|---------|
| Python+deps baseline               |   101.1 MB | —              | —       |
| Traced memory (current)            |  267.19 MB | —              | —       |
| Traced memory (peak)               |  393.53 MB | —              | —       |
| Post-load RSS                      |   438.4 MB | —              | —       |
| RSS diff                           |   337.3 MB | —              | —     |
| movies                             |     3.9 MB | —              | —       |
| catalog                            |    13.7 MB | —              | —       |
| hybrid.content.user_profiles       |    29.8 MB | (195544, 20)   | float64 |
| hybrid.content.rated_indices_      |   121.0 MB | (31720336,)    | int32   |
| hybrid.collab.U                    |    74.6 MB | (195544, 50)   | float64 |
| hybrid.collab.Vh                   |    11.6 MB | (50, 30521)    | float64 |
| hybrid.collab.rated_indices_       |   121.0 MB | (31720336,)    | int32   |
| popularity.scores                  |     0.2 MB | (30521,)       | float64 |


## After scikit dep removed 
_Run: 2026-04-27 15:56:48_

| Component                          | Size      | Shape          | Dtype   |
|------------------------------------|-----------|----------------|---------|
| Python+deps baseline               |   101.0 MB | —              | —       |
| Traced memory (current)            |  263.11 MB | —              | —       |
| Traced memory (peak)               |  389.46 MB | —              | —       |
| Post-load RSS                      |   433.6 MB | —              | —       |
| RSS diff                           |   332.5 MB | —              | —     |
| movies                             |     3.9 MB | —              | —       |
| catalog                            |     9.6 MB | —              | —       |
| hybrid.content.user_profiles       |    29.8 MB | (195544, 20)   | float64 |
| hybrid.content.rated_indices_      |   121.0 MB | (31720336,)    | int32   |
| hybrid.collab.U                    |    74.6 MB | (195544, 50)   | float64 |
| hybrid.collab.Vh                   |    11.6 MB | (50, 30521)    | float64 |
| hybrid.collab.rated_indices_       |   121.0 MB | (31720336,)    | int32   |
| popularity.scores                  |     0.2 MB | (30521,)       | float64 |

## float32 for factors
_Run: 2026-04-27 16:21:30_

| Component                          | Size      | Shape          | Dtype   |
|------------------------------------|-----------|----------------|---------|
| Python+deps baseline               |   100.7 MB | —              | —       |
| Traced memory (current)            |  205.08 MB | —              | —       |
| Traced memory (peak)               |  331.42 MB | —              | —       |
| Post-load RSS                      |   375.9 MB | —              | —       |
| RSS diff                           |   275.2 MB | —              | —     |
| movies                             |     3.9 MB | —              | —       |
| catalog                            |     9.6 MB | —              | —       |
| hybrid.content.user_profiles       |    14.9 MB | (195544, 20)   | float32 |
| hybrid.content.rated_indices_      |   121.0 MB | (31720336,)    | int32   |
| hybrid.collab.U                    |    37.3 MB | (195544, 50)   | float32 |
| hybrid.collab.Vh                   |     5.8 MB | (50, 30521)    | float32 |
| hybrid.collab.rated_indices_       |   121.0 MB | (31720336,)    | int32   |
| popularity.scores                  |     0.2 MB | (30521,)       | float64 |

## int32 -> unint16, up to 2**16 - 1 movies, we don't need sign for indices
_Run: 2026-04-27 18:33:53_

| Component                          | Size      | Shape          | Dtype   |
|------------------------------------|-----------|----------------|---------|
| Python+deps baseline               |   100.6 MB | —              | —       |
| Traced memory (current)            |  144.58 MB | —              | —       |
| Traced memory (peak)               |  210.42 MB | —              | —       |
| Post-load RSS                      |   315.2 MB | —              | —       |
| RSS diff                           |   214.6 MB | —              | —     |
| movies                             |     3.9 MB | —              | —       |
| catalog                            |     9.6 MB | —              | —       |
| hybrid.content.user_profiles       |    14.9 MB | (195544, 20)   | float32 |
| hybrid.content.rated_indices_      |    60.5 MB | (31720336,)    | uint16  |
| hybrid.collab.U                    |    37.3 MB | (195544, 50)   | float32 |
| hybrid.collab.Vh                   |     5.8 MB | (50, 30521)    | float32 |
| hybrid.collab.rated_indices_       |    60.5 MB | (31720336,)    | uint16  |
| popularity.scores                  |     0.2 MB | (30521,)       | float64 |


## float32 -> float16, vaildated top10 intersection same for ~99.4% simulations
_Run: 2026-04-27 19:46:45_

| Component                          | Size      | Shape          | Dtype   |
|------------------------------------|-----------|----------------|---------|
| Python+deps baseline               |   101.1 MB | —              | —       |
| Traced memory (current)            |  115.56 MB | —              | —       |
| Traced memory (peak)               |  181.40 MB | —              | —       |
| Post-load RSS                      |   286.8 MB | —              | —       |
| RSS diff                           |   185.7 MB | —              | —     |
| movies                             |     3.9 MB | —              | —       |
| catalog                            |     9.6 MB | —              | —       |
| hybrid.content.user_profiles       |     7.5 MB | (195544, 20)   | float16 |
| hybrid.content.rated_indices_      |    60.5 MB | (31720336,)    | uint16  |
| hybrid.collab.U                    |    18.6 MB | (195544, 50)   | float16 |
| hybrid.collab.Vh                   |     2.9 MB | (50, 30521)    | float16 |
| hybrid.collab.rated_indices_       |    60.5 MB | (31720336,)    | uint16  |
| popularity.scores                  |     0.2 MB | (30521,)       | float64 |

## checked potential modification of loaded objects before proceeding
## after mmap load 
_Run: 2026-04-27 20:06:15_

| Component                          | Size      | Shape          | Dtype   |
|------------------------------------|-----------|----------------|---------|
| Python+deps baseline               |   101.3 MB | —              | —       |
| Traced memory (current)            |   18.63 MB | —              | —       |
| Traced memory (peak)               |   36.67 MB | —              | —       |
| Post-load RSS                      |   190.4 MB | —              | —       |
| RSS diff                           |    89.1 MB | —              | —     |
| RSS after gc.collect (1st)         |   203.3 MB | —              | —       |
| RSS after gc.collect (100th)       |   223.4 MB | —              | —       |
| movies                             |     3.9 MB | —              | —       |
| catalog                            |     9.6 MB | —              | —       |
| hybrid.content.user_profiles       |     0.1 MB | (195544, 20)   | float16 |
| hybrid.content.rated_indices_      |     0.6 MB | (31720336,)    | uint16  |
| hybrid.collab.U                    |     0.2 MB | (195544, 50)   | float16 |
| hybrid.collab.Vh                   |     0.0 MB | (50, 30521)    | float16 |
| hybrid.collab.rated_indices_       |     0.6 MB | (31720336,)    | uint16  |
| popularity.scores                  |     0.2 MB | (30521,)       | float64 |


## after mmap load v2, one more run 
_Run: 2026-04-27 20:14:32_

| Component                          | Size      | Shape          | Dtype   |
|------------------------------------|-----------|----------------|---------|
| Python+deps baseline               |   100.5 MB | —              | —       |
| Traced memory (current)            |   18.63 MB | —              | —       |
| Traced memory (peak)               |   36.65 MB | —              | —       |
| Post-load RSS                      |   189.6 MB | —              | —       |
| RSS diff                           |    89.1 MB | —              | —     |
| RSS after gc.collect (1st)         |   200.7 MB | —              | —       |
| RSS after gc.collect (100th)       |   225.3 MB | —              | —       |
| movies                             |     3.9 MB | —              | —       |
| catalog                            |     9.6 MB | —              | —       |
| hybrid.content.user_profiles       |     0.1 MB | (195544, 20)   | float16 |
| hybrid.content.rated_indices_      |     0.6 MB | (31720336,)    | uint16  |
| hybrid.collab.U                    |     0.2 MB | (195544, 50)   | float16 |
| hybrid.collab.Vh                   |     0.0 MB | (50, 30521)    | float16 |
| hybrid.collab.rated_indices_       |     0.6 MB | (31720336,)    | uint16  |
| popularity.scores                  |     0.2 MB | (30521,)       | float64 |
