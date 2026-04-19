import json
import os

from dotenv import load_dotenv
from google import genai
from metaflow import FlowSpec, Parameter, step


class Assignment13(FlowSpec):
    """LLM generates a student list; foreach processes each student.

    Flow: start -> process_student (foreach) -> join -> end
    The 'start' step asks Gemini to produce a JSON list of students.
    Each foreach branch uppercases the name and adds a score bonus.
    The join step aggregates all updated students and totals the scores.
    """

    bonus = Parameter(
        "bonus",
        help="Score bonus to add to each student",
        default=10,
        type=int,
    )

    @step
    def start(self):
        """Ask the LLM for a list of students, then fan out via foreach."""
        load_dotenv()
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        prompt = (
            "Return a JSON array of 5 students. "
            "Each student is an object with exactly two keys: "
            "'name' (string) and 'score' (integer between 50 and 100). "
            "Reply with only the raw JSON array, no markdown or explanation."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        raw = response.text.strip()
        # Strip markdown code fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        self.students = json.loads(raw.strip())

        print("Students from LLM:")
        for s in self.students:
            print(f"  {s}")

        self.next(self.process_student, foreach="students")

    @step
    def process_student(self):
        """Uppercase the name and apply the score bonus."""
        student = self.input
        self.updated = {
            "name": student["name"].upper(),
            "score": student["score"] + self.bonus,
        }
        print(f"  {student} -> {self.updated}")
        self.next(self.join)

    @step
    def join(self, inputs):
        """Collect all updated students and compute aggregate score."""
        self.updated_students = [i.updated for i in inputs]
        self.total_score = sum(s["score"] for s in self.updated_students)
        self.merge_artifacts(inputs, exclude=["updated"])
        self.next(self.end)

    @step
    def end(self):
        """Print the final updated list and aggregate score."""
        print("\n--- Updated students ---")
        for s in self.updated_students:
            print(f"  {s['name']}: {s['score']}")
        print(f"\nTotal score: {self.total_score}")
        print(f"Average score: {self.total_score / len(self.updated_students):.1f}")


if __name__ == "__main__":
    Assignment13()
