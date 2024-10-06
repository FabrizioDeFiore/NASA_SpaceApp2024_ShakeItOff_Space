using UnityEngine;

public class AnimControl : MonoBehaviour
{
    public Animator animator;              // Reference to the Animator component
    public bool startAnimation;            // Boolean to trigger the animation
    public bool resetAnimation;            // Boolean to trigger the reset
    public string animationClipName;       // The name of the animation clip
    public float newDuration;              // New duration for the animation (in seconds)
    public float progress;
    public bool animating;
    private float originalDuration;
    private float elapsedTime;

    private void Update()
    {
        
        // Check if the startAnimation boolean is set to true
        if (startAnimation)
        {
            StartAnimation();              // Play the animation
            startAnimation = false;        // Prevent restarting the animation multiple times
            animating = true;
        }

        // Update the progress of the animation duration
        if (animating)
        {
            elapsedTime += Time.deltaTime;
            progress = Mathf.Clamp01(elapsedTime / newDuration) * 100;
        }
        else
        {
            progress = 0.0f; 
            elapsedTime = 0;
        }
    }

    public bool IsAnimationPlaying()
    {
        if (animator != null)
        {
            AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
            return stateInfo.normalizedTime % 1 > 0; // Check if the animation is playing
        }
        return false;
    }

    public void ResetAnimation()
    {
        animating = false;
        if (animator != null & progress >= 0.2f)
        {
            animator.SetTrigger("Reset");
        }
    }

    // Function to start playing the animation on command
    public void StartAnimation()
    {
        animating = true;
        // Ensure the Animator component is assigned
        if (animator == null)
        {
            return;
        }

        AdjustAnimationDuration(newDuration);
        animator.SetTrigger("Activate");
    }

    public void AdjustAnimationDuration(float targetDuration)
    {
        if (animator == null)
        {
            Debug.LogError("Animator component is not assigned!");
            return;
        }

        AnimationClip clip = GetAnimationClip(animator, animationClipName);
        Debug.Log($"Animation clip: {clip}");
        if (clip != null)
        {
            originalDuration = clip.length;
            float speedMultiplier = originalDuration / targetDuration;
            animator.speed = speedMultiplier;
            newDuration = targetDuration;
            elapsedTime = 0; // Reset elapsed time
        }
    }

    private AnimationClip GetAnimationClip(Animator anim, string clipName)
    {
        foreach (AnimationClip clip in anim.runtimeAnimatorController.animationClips)
        {
            if (clip.name == clipName)
            {
                return clip;
            }
        }
        return null;
    }
}
