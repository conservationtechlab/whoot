import os
from dotenv import load_dotenv
from label_studio_sdk import LabelStudio

class LabelStudioSetup():
    """Sets up a Label Studio project for annotation."""

    def __init__(self, 
                 template_path: str = "template.xml",
                 currnet_project = None):
        """Initialize the Label Studio client and create a project."""
        load_dotenv()
        LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
        LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

        if LABEL_STUDIO_URL is None or LABEL_STUDIO_API_KEY is None:
            raise ValueError("LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY must be set in the .env file.")

        self.client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)


        with open(template_path, "r") as f:
            self.label_config =  f.read()

        self.current_project = currnet_project

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

        print("Project ID:", project.id)
        
        # Associate this class instance with the created project
        self.current_project = project

        return project


