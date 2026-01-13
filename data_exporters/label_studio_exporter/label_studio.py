from http import client
import os
import requests
from dotenv import load_dotenv
from label_studio_sdk import LabelStudio
import tqdm
from label_studio_sdk.label_interface.objects import PredictionValue, AnnotationValue
import datasets

class LabelStudioSetup():
    """Sets up a Label Studio project for annotation.
    
    When submoduling, primarly do so for diffrent labeling templates. In particular,
    - apply_audio_template
    - default_template_annotation_style

    These will be template spefific. Currently, they mirror the template found in
    data_exporters/label_studio_exporter/template.xml
    """

    def __init__(self, current_project = None):
        """Initialize the Label Studio client and create a project."""
        load_dotenv()
        LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
        LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

        
        if LABEL_STUDIO_URL is None or LABEL_STUDIO_API_KEY is None:
            raise ValueError("LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY must be set in the .env file.")

        self.client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

        if current_project is not None:
            self.current_project = self.client.projects.get(current_project)
            print("Project ID:", self.current_project.id, "\t Project Name:", self.current_project.title)
            input("Double check this, this script can take destructive actions. Press Enter to continue...")
        
        self.api_key = LABEL_STUDIO_API_KEY
        self.base_url = LABEL_STUDIO_URL

    def create_project(self, title: str = "Whoot Audio Annotation Project"):
        """Create a new project in Label Studio.

        Args:
            title (str): The title of the project.

        Returns:
            project: The created Label Studio project.
        """
        project = self.client.projects.create(
            title=title,
            label_config=self.label_config
        )

        print("Project ID:", project.id, project.title)
        input("Double check this, this script can take destructive actions. Press Enter to continue...")
        
        # Associate this class instance with the created project
        self.current_project = project

        return project

    def apply_audio_template(self, class_names: list = None):
        """Apply a default audio annotation template.
        
        Args:
            class_names (list): List of class names for labeling.
        """
        
        if self.current_project is None:
            raise ValueError("No current project set. Please create a project first.")
        
        audio_template = """
        <View>
        <Labels name="labels" toName="audio">
        """

        for class_name in class_names:
            audio_template += f'<Label value="{class_name}"/>\n'

        audio_template += """
        </Labels>
        <Audio name="audio" value="$audio" decoder="ffmpeg" spectrogram="true" height="500"/>
        """      

        print("Applying audio annotation template to project ID:", self.current_project.id)
        print(audio_template)  

        self.set_label_interface(template=audio_template)
        self.xml_template = audio_template

    def apply_custom_template(self, template_path: str):
        """Apply a custom annotation template from a file.

        Args:
            template_path (str): Path to the XML template file.
        """
        if self.current_project is None:
            raise ValueError("No current project set. Please create a project first.")

        with open(template_path, "r") as f:
            custom_template = f.read()

        print("Applying custom annotation template to project ID:", self.current_project.id)
        print(custom_template)
        self.set_label_interface(template=custom_template)
        self.xml_template = custom_template

    def set_label_interface(self, template: str):
        req = self.client.projects.update(
            id=self.current_project.id,
            label_config=template
        )
        print(req)

    def get_files(self, ls_file_parent: str = None):
        """Retrieve all files in the current project.

        Returns:
            list: List of files in the project.
        """
        if self.current_project is None:
            raise ValueError("No current project set. Please create a project first.")

        files = []
        ids = []
        response = self.client.tasks.list(project=self.current_project.id)
        for item in response:
            audio_path = item.data["audio"]
            audio_path = audio_path.replace(self.base_url, "")
            if ls_file_parent is not None:
                audio_path = audio_path.replace(ls_file_parent, "")
            files.append(audio_path)
            ids.append(item.id)
        return {"files": files, "ids": ids}

    def add_labels_to_task(self, task_id: int, result: list, prediction=True):
        """Update all tasks in the current project.

        Args:
            tasks (list): List of task dictionaries to update.
        """
        if self.current_project is None:
            raise ValueError("No current project set. Please create a project first.")

        if prediction:
            self.client.predictions.update(
                id=self.current_project.id,
                task=task_id,
                result=result,
            )
        else:
            self.client.annotations.update(
                id=self.current_project.id,
                task=task_id,
                ground_truth=True,
                result=result,
            )


    def default_template_annotation_style(
            self,
            id: int,
            offset: float,
            duration: float,
            label: str,
            file_path: str,
            prediction: bool = False
        ):

        label_type = "annotations"
        if prediction:
            label_type = "predictions"

        return {
            "id": id,
            f"{label_type}":
                {
                    "from_name": "labels",
                    "to_name": "audio",
                    "type": "labels",
                    "value": {
                        "start": offset,
                        "end": offset + duration,
                        "labels": [label]
                    }
                }
            ,
            "data": {
                "audio": file_path
            }
        }

    def update_tasks_in_ls(self, ds: datasets.Dataset, ls_file_parent: str, is_model_prediction=True):
        li = self.current_project.get_label_interface()
        files = self.get_files(ls_file_parent=ls_file_parent)["files"]
        task_ids = self.get_files(ls_file_parent=ls_file_parent)["ids"]

        datasets.disable_progress_bars()

        for i in tqdm.tqdm(range(len(files)), desc="Updating tasks in project: {}".format(self.current_project.title)):
            id = task_ids[i]
            file_ds = ds.filter(lambda x: x['audio']['path'] == files[i])

            # Our custom configuration of datasets allow for segmentation labels in audio :)
            # This checks for it
            if "offset" in file_ds[0]["audio"]:
                offset = file_ds[0]["audio"]["offset"]
                duration = file_ds[0]["audio"]["duration"]
            else:
                offset = 0.0
                duration = 1.0


            file_ds = file_ds.map(lambda x: self.default_template_annotation_style(
                id, offset, duration, x['labels'], x['audio']['path'], prediction=is_model_prediction),)

            if is_model_prediction:
                prediction = PredictionValue(
                    # # Tag predictions with specific model version
                    model_version='my_model_v1',
                    result=file_ds["predictions"]
                )
                self.client.predictions.create(task=id, **prediction.model_dump())
            else:
                annotations = AnnotationValue(
                    # Define your labels here
                    result=file_ds["annotations"]
                )
                self.client.annotations.create(task=id, **annotations.model_dump())
        datasets.enable_progress_bars()