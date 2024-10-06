using UnityEngine;

public class LineProgression : MonoBehaviour
{
    public LineRenderer lineRenderer; // Assign LineRenderer via Inspector
    public float startX = -10.0f;     // Starting x-position (left side of the graph)
    public float endX = 10.0f;        // Ending x-position (right side of the graph)
    public float lineHeight = 5.0f;   // Height of the line (y-values)
    public float duration = 5.0f;     // Duration for the line to fully slide across
    private float elapsedTime = 0f;   // Timer for the animation

    void Start()
    {
        // Initialize the LineRenderer's start and end points to the left of the graph
        lineRenderer.positionCount = 2; // Two points for the line
        lineRenderer.SetPosition(0, new Vector3(startX, -lineHeight, 0)); // Starting point
        lineRenderer.SetPosition(1, new Vector3(startX, lineHeight, 0));  // Ending point (y-axis)
    }

    void Update()
    {
        // Increment the timer
        elapsedTime += Time.deltaTime;

        // Calculate how far along the line should be based on time elapsed
        float t = Mathf.Clamp01(elapsedTime / duration);

        // Calculate the current x-position based on linear interpolation (lerp)
        float currentX = Mathf.Lerp(startX, endX, t);

        // Update the LineRenderer's position to reflect the new x-position
        lineRenderer.SetPosition(0, new Vector3(currentX, -lineHeight, 0)); // Bottom of the line
        lineRenderer.SetPosition(1, new Vector3(currentX, lineHeight, 0));  // Top of the line
    }
}
