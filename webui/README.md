# SEARL WebUI

This is the Web User Interface for SEARL, built with Gradio. The interface provides an intuitive way to manage all aspects of reinforcement learning experiments.

## Feature Modules

The WebUI includes the following five main modules:

1. **Data Processing** – For managing and processing experiment data
2. **Tool Definition** – For defining and managing experiment tools
3. **Environment Definition** – For configuring and managing experiment environments
4. **Training & Deployment** – For training models and deploying experiments
5. **Project Management** – For managing experiment projects and resources

## Installation

1. Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

The application will start at http://localhost:7860.

## Development Notes

- Each feature module is implemented as a separate tab in `app.py`
- The interface is built using Gradio's Blocks API
- All components support real-time updates and interaction

## Notes

- Ensure all necessary dependencies are installed before running the application
- The default port is 7860, which can be modified in `app.py`
- Debug mode is enabled during development; please disable it for production deployment
