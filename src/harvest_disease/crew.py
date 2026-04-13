import os
import yaml
from src.harvest_disease.tools.custom_tool import HarvestWheatDiseaseDetectionTool, solutionWebsiteSearchTool

class CrewConfig:
    def __init__(self, agents, tasks):
        self.agents = agents
        self.tasks = tasks

    @classmethod
    def from_yaml(cls, agents_path=None, tasks_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        agents_full_path = os.path.join(base_dir, "config", "agents.yaml")
        tasks_full_path = os.path.join(base_dir, "config", "tasks.yaml")
        
        with open(agents_full_path, 'r') as a:
            agents = yaml.load(a, Loader=yaml.FullLoader)
        with open(tasks_full_path, 'r') as t:
            tasks = yaml.load(t, Loader=yaml.FullLoader)
        return cls(agents, tasks)


class Crew:
    def __init__(self, agents, tasks, verbose=False):
        self.agents = agents
        self.tasks = tasks.get('tasks', [])
        self.verbose = verbose
        self.outputs = {}

    def kickoff(self):
        self.execute_tasks()
        return self.outputs

    def resolve_input(self, input_data):
        if isinstance(input_data, dict):
            resolved = {}
            for key, value in input_data.items():
                if isinstance(value, str) and value.startswith("{{") and "output" in value:
                    parts = value.strip("{{}}").split(".")
                    task_name = parts[0]
                    output_key = parts[-1]
                    resolved[key] = self.outputs.get(task_name, {}).get(output_key, "UNKNOWN")
                else:
                    resolved[key] = value
            return resolved
        return input_data

    def execute_tasks(self):
        if self.verbose:
            print("📋 Agents chargés :", list(self.agents.keys()))
            print("📋 Tâches à exécuter :", [task['name'] for task in self.tasks])

        for task in self.tasks:
            name = task['name']
            agent_name = task['agent']
            tools = task.get("tools", [])
            input_data = self.resolve_input(task.get("input", {}))
            output = {}

            print(f"\n🚀 Exécution de la tâche: {name}")
            print(f"👤 Agent assigné: {agent_name}")
            print(f"🧰 Outils: {tools}")
            print(f"📥 Données d'entrée: {input_data}")

            for tool in tools:
                if tool == "HarvestWheatDiseaseDetectionTool":
                    classifier = HarvestWheatDiseaseDetectionTool()

                    predictions = []
                    image_paths = input_data.get("image_paths", [])
                    for img_path in image_paths:
                        result = classifier._run(image_path=img_path)
                        s = result.split("(")[0]
                        predictions.append({"image": img_path, "disease": s})
                    output = {"predictions": predictions}

                elif tool == "solutionWebsiteSearchTool":
                    searcher = solutionWebsiteSearchTool()
                    previous_results = self.outputs.get("classify_harvest_disease_task", {}).get("predictions", [])
                    enriched_results = []
                    for prediction in previous_results:
                        disease_name = prediction["disease"].strip().split(" (")[0]
                        image = prediction.get("image", "N/A")

                        if disease_name == "Wheat_healthy":
                            solution = searcher.search(disease_name, image)
                            prediction["solution"] = "No treatment needed. The wheat is healthy."
                        else:
                            solution = searcher.search(disease_name, image)
                            prediction["solution"] = solution

                        enriched_results.append(prediction)

                    output = {"predictions": enriched_results}

            self.outputs[name] = output
            print(f"✅ Résultat de la tâche {name}: {output}")
            print("-" * 50)


# ==== Exécution ====
config = CrewConfig.from_yaml()

crew = Crew(
    agents=config.agents,
    tasks=config.tasks,
    verbose=True
)
