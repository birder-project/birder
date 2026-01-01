# Adversarial

The `adversarial` tool allows you to generate and visualize adversarial examples for bird classification models. This tool implements several attack methods that create imperceptible perturbations to images, causing models to misclassify them. It is useful for evaluating model robustness and understanding model vulnerabilities.

## Usage

```sh
python -m birder.tools adversarial [OPTIONS] IMAGE_PATH
```

To see all available options and get detailed help, run:

```sh
python -m birder.tools adversarial --help
```

This will display a comprehensive list of all options and their descriptions, ensuring you have access to the full functionality of the `adversarial` tool.

## Description

The tool supports multiple adversarial attack methods:

- **FGSM (Fast Gradient Sign Method)**: A single-step attack that generates adversarial examples by adding perturbations in the direction of the gradient
- **PGD (Projected Gradient Descent)**: An iterative attack that repeatedly applies small perturbations while staying within the allowed perturbation budget
- **DeepFool**: An iterative attack that finds the minimum perturbation needed to change the model's prediction
- **SimBA**: A black-box attack that only requires model outputs and does not need gradient information

Key features include:

- **Targeted and untargeted attacks**: Specify a target class for targeted attacks, or leave empty for untargeted attacks that simply aim to change the prediction
- **Configurable perturbation budgets**: Control the maximum allowed perturbation in pixel space
- **Visual comparison**: Side-by-side display of original and adversarial images with predictions and confidence scores
- **Attack success metrics**: Reports whether the attack succeeded and the number of model queries used

## Notes

- All perturbations are specified in pixel space (range [0, 1]) for intuitive control across different models and input normalizations
- The tool automatically handles the conversion between pixel space and the model's input normalization
- Some attack methods require gradient information and may have limitations with certain network architectures or activation functions (for example, using `F.relu` instead of `nn.ReLU` may affect gradient-based attacks)
- Black-box attacks like SimBA do not rely on gradients and can be applied more broadly, but typically require more model queries
- For iterative attacks, increasing the number of steps generally improves attack success but requires more computation

For more detailed information about each option and its usage, refer to the help output of the tool.
