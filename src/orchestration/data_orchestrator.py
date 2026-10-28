
class DataOrchestrator:

    def __init__(self, stages: dict):
        self.stages = stages

    def run(self, stage: str, **kwargs) -> dict:

        if stage not in self.stages:
            raise ValueError(
                f"La etapa '{stage}' no está registrada."
            )

        pipeline = self.stages[stage]

        return pipeline(**kwargs)
