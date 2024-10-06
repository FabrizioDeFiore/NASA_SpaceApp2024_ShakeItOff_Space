using UnityEngine;
using TMPro;

public class DataScript : MonoBehaviour
{
    public TMP_Text magnitudeText;  // Reference to the TextMeshPro - Text component
    public TMP_Text velocityText;  // Reference to the TextMeshPro - Text component
    public TMP_Text filenameText;  // Reference to the TextMeshPro - Text component
    public string filename;
    private string newText;        // The new text to set
    private string newText1;        // The new text to set
    private string newText2;        // The new text to set
    public double magnitude;
    public double velocity;

    private void Start()
    {
        newText = "Magnitude: " + magnitude;
        newText1 = "Velocity: " + velocity + " (m/s)";
    }

    private void Update()
    {
        newText = "Magnitude: " + magnitude;
        newText1 = "Velocity: " + velocity + " (m/s)";
        newText2 = filename;
        // Check if the newText is not empty and the magnitudeText component is assigned
        if (!string.IsNullOrEmpty(newText) && magnitudeText != null)
        {
            UpdateText(newText);
            newText = string.Empty;  // Clear the newText to prevent continuous updates
        }
        if (!string.IsNullOrEmpty(newText1) && velocityText != null)
        {
            UpdateText1(newText1);
            newText1 = string.Empty;  // Clear the newText to prevent continuous updates
        }
        if (!string.IsNullOrEmpty(newText2) && filenameText != null)
        {
            UpdateFilename(newText2);
            newText2 = string.Empty;  // Clear the newText to prevent continuous updates
        }
    }

    // Function to update the text in the TextMeshPro component
    public void UpdateText(string text)
    {
        magnitudeText.text = text;     
    }
    public void UpdateText1(string text)
    {
        velocityText.text = text;       
    }
    public void UpdateFilename(string text)
    {
        filenameText.text = text;       
    }
}
