using UnityEngine;
using TMPro;

public class DataScript : MonoBehaviour
{
    public TMP_Text textMeshPro;  // Reference to the TextMeshPro - Text component
    public TMP_Text textMeshPro1;  // Reference to the TextMeshPro - Text component
    private string newText;        // The new text to set
    private string newText1;        // The new text to set
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
        // Check if the newText is not empty and the textMeshPro component is assigned
        if (!string.IsNullOrEmpty(newText) && textMeshPro != null)
        {
            UpdateText(newText);
            newText = string.Empty;  // Clear the newText to prevent continuous updates
        }
        if (!string.IsNullOrEmpty(newText1) && textMeshPro1 != null)
        {
            UpdateText1(newText1);
            newText1 = string.Empty;  // Clear the newText to prevent continuous updates
        }
    }

    // Function to update the text in the TextMeshPro component
    public void UpdateText(string text)
    {
        textMeshPro.text = text;     
    }
    public void UpdateText1(string text)
    {
        textMeshPro1.text = text;       
    }
}
