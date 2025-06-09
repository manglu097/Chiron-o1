class Step:
    def __init__(self, step_text="", prefix_steps="", parent=None, score=0.0, generated_by=None):
        """
        Initializes a step node in the reasoning tree.

        Args:
            step_text: Text of the current reasoning step (e.g., "### Step 1: ..."). Typically starts with '###'.
            prefix_steps: Concatenated text of all preceding steps (parent's text).
            parent: Parent step node.
            score: Score of this step (based on student model evaluation).
            generated_by: Teacher model that generated this step.
        """
        self.step_text = step_text
        self.parent = parent
        self.children = []
        self.score = score  # Score from student evaluation
        self.prefix_steps = prefix_steps  # All steps before this node
        self.generated_by = generated_by  # Tracks which teacher generated this step
        self.depth = parent.depth + 1 if parent else 0
        
        if parent:
            separator = "\n" if self.prefix_steps else ""
            self.text = self.prefix_steps + separator + self.step_text
        else:  # Root node
            self.text = prefix_steps  

    def is_terminal(self):
        """
        Checks if the current step is a terminal step (i.e., has no children).
        """
        return len(self.children) == 0

    def add_child_step(self, step_text, score=0.0, generated_by=None):
        """
        Adds a child step to the current step.
        The child step's prefix_steps is the current step's full text.
        """
        if step_text is None:
            print("Warning: Attempted to add a child step with None step_text.")
            return None
        child_step = Step(step_text=step_text, prefix_steps=self.text, parent=self, score=score, generated_by=generated_by)
        self.children.append(child_step)
        return child_step

    def get_full_reasoning(self):
        """
        Returns the full reasoning chain text for this step.
        """
        return self.text

    def get_step_path(self):
        """
        Returns the list of steps from the root to the current step.
        """
        step = self
        path = []
        while step:
            path.append(step)
            step = step.parent
        return path[::-1]  # Return from root step