import wandb

# Initialize the API
api = wandb.Api()

# Define the entity name
entity_name = 'martinpreiss'

# Get all projects for the entity
projects = api.projects(entity=entity_name)

# Iterate over all projects
for project in projects:
    project_name = project.name
    print(f"Processing project: {project_name}")

    # Get all runs for the project
    runs = api.runs(f"{entity_name}/{project_name}")

    # Iterate over all runs
    for run in runs:
        run_name = run.name
        if run_name.endswith('cf_'):
            new_name = run_name[:-3] + 'ct_'
        elif run_name.endswith('ct_'):
            new_name = run_name[:-3] + 'cf_'
        else:
            continue

        # Rename the run
        run.name = new_name
        run.update()
        print(f"Renamed run {run.id} from {run_name} to {new_name}")

print("Renaming completed.")