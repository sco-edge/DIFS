# DIFS
[comment]: # (This is the Diffusion Model Inference Service. The work takes inspiration from the automated model-less
systems like "INFaaS: Automated Model-less Inference Serving" which simplify the model specification
process for developers. Such models speed up model variant generation by relying on a repository of
already trained models whiles optimizing the navigational search through the tradeoff space. Our system
is based on the Stable Diffusion Model.)

DIFS (Diffusion Model Inference Service) is a specialized, local inference platform designed to streamline
the deployment and serving of text-to-image diffusion models, with a primary focus on Stable Diffusion and
its variants. Inspired by automated, model-less inference systems such as INFaaS ("Automated Model-less
Inference Serving," OSDI 2020), DIFS eliminates the complexity of manual model specification and resource
management, enabling developers to focus on prompt engineering and creative output rather than infrastructure tuning.
Like INFaaS, DIFS adopts a model-less abstraction: users specify high-level task requirements (e.g., desired
latency, output resolution, or quality level) rather than exact model variants. The system then automatically selects and routes requests to the most suitable pre-registered diffusion model from a local repository, navigating the inherent trade-off space between speed, memory usage, and visual fidelity.
Key differentiators and contributions of DIFS include:

* Local, Single-Machine Optimization:
- Unlike the original cloud-centric INFaaS (which relied on AWS EC2/S3), DIFS is fully adapted for standalone execution
   on resource-constrained hardware, including machines with a single consumer-grade GPU. All AWS dependencies have been
    removed, and model/data loading uses local file paths.
    
* Diffusion-Specific Architecture Support:
- Built around the iterative denoising process of diffusion models, DIFS enables dynamic control over inference parameters
   (e.g., number of steps, guidance scale) and supports efficient memory techniques (attention slicing, sequential offloading)
   to maximize performance on limited VRAM.
   
* Extensible Scaling Potential:
- While optimized for single-GPU setups, the retained INFaaS worker-master design allows future extensions to multi-GPU vertical
   scaling or multi-node horizontal scaling for high-throughput scenarios.
   
* Privacy and Cost Efficiency:
- By operating entirely offline, DIFS ensures data privacy and eliminates cloud costs, making it ideal for on-premises creative
  tools, research prototyping, and edge deployment.

At its core, DIFS leverages the Stable Diffusion family (e.g., v1.5, SDXL, Turbo, and LoRA-adapted variants) as the primary generative
backbone, registered as profiled model variants within the system. This combination of automated variant selection, local execution, and
diffusion-aware scheduling positions DIFS as a practical, accessible platform for deploying state-of-the-art text-to-image generation in
real-world, resource-constrained environments.

In essence, DIFS transforms the sophisticated model-serving concepts of INFaaS into a dedicated, diffusion-first inference service—bridging
the gap between cutting-edge generative AI and deployable, user-friendly systems.
