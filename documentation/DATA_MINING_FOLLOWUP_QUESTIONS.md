# Data Mining Follow-up Questions Guide

This document provides a comprehensive list of follow-up questions that users can ask about data mining results. These questions are designed to be answered by the AI system using the structured data from rule mining and distance correlation analyses.

## Distance Correlation Follow-up Questions

### High-Level Analysis Questions
1. **"Which chiplet type has the strongest overall impact on performance?"**
2. **"What's the relationship between GPU count and energy consumption?"**
3. **"How does the number of Attention chiplets affect execution time?"**
4. **"Which chiplet type shows the most surprising correlation pattern?"**
5. **"Are there any chiplet types that have minimal impact on performance?"**

### Specific Correlation Questions
6. **"Why does GPU have such a high correlation with energy consumption?"**
7. **"What explains the correlation between Sparse chiplets and execution time?"**
8. **"How do the correlations change if we focus only on Pareto-optimal designs?"**
9. **"Which chiplet type has the most balanced impact on both energy and time?"**
10. **"What's the significance of the correlation value for Convolution chiplets?"**

### Design Decision Questions
11. **"Based on these correlations, what's the optimal GPU count for energy efficiency?"**
12. **"How should I adjust my chiplet allocation to minimize execution time?"**
13. **"What's the trade-off between Attention and Sparse chiplets for my objective?"**
14. **"Which chiplet type should I prioritize if I want to optimize both metrics?"**
15. **"How do these correlations inform my design constraints?"**

### Comparative Questions
16. **"How do these correlations compare to typical chiplet design patterns?"**
17. **"What's different about the correlation patterns in this trace compared to others?"**
18. **"Are these correlations consistent across different optimization runs?"**
19. **"How do the energy vs time correlations differ for each chiplet type?"**
20. **"What's the relative importance of each chiplet type for my specific goal?"**

## Rule Mining Follow-up Questions

### Rule Pattern Questions
21. **"What's the most common pattern in the optimal designs?"**
22. **"Which rule has the highest confidence and what does it mean?"**
23. **"What do the lift values tell us about rule significance?"**
24. **"Are there any conflicting rules in the results?"**
25. **"Which rule combinations appear most frequently in Pareto-optimal designs?"**

### Confidence Analysis Questions
26. **"What does a confidence of 0.85 from features to Pareto front mean?"**
27. **"How reliable are these rules for predicting optimal designs?"**
28. **"Which rules have the highest lift and why is that important?"**
29. **"What's the difference between conf(f→p) and conf(p→f)?"**
30. **"How should I interpret the confidence values for design decisions?"**

### Specific Rule Questions
31. **"What does the rule 'High GPU AND Low Attention' tell us?"**
32. **"Why is the rule about Sparse chiplets so significant?"**
33. **"How do the rules change if we look at different Pareto ranks?"**
34. **"What's the practical meaning of the 'Medium Convolution' rule?"**
35. **"Which rules are most actionable for my design optimization?"**

### Design Application Questions
36. **"How should I apply these rules to create a new design?"**
37. **"Which rule should I prioritize for energy optimization?"**
38. **"What's the best chiplet combination based on these rules?"**
39. **"How do these rules help me understand design trade-offs?"**
40. **"Which rules are most relevant for my specific trace?"**

### Comparative Rule Questions
41. **"How do these rules compare to general chiplet design principles?"**
42. **"What's unique about the rules found in this analysis?"**
43. **"How do the rules differ between energy-focused and time-focused optimization?"**
44. **"Are there any rules that contradict common design wisdom?"**
45. **"How do these rules apply to different types of workloads?"**

## Cross-Analysis Questions

### Integration Questions
46. **"How do the correlation results align with the rule mining findings?"**
47. **"What insights emerge when combining both analyses?"**
48. **"Are there any contradictions between correlation and rule patterns?"**
49. **"How do both analyses inform the same design decision?"**
50. **"What's the complete picture from both distance correlation and rule mining?"**

### Practical Application Questions
51. **"Based on both analyses, what's my optimal chiplet configuration?"**
52. **"How should I balance the insights from correlations vs rules?"**
53. **"What's the most important takeaway from both analyses?"**
54. **"How do I translate these findings into actionable design changes?"**
55. **"What's the next step in my optimization process?"**

## Technical Deep-Dive Questions

### Statistical Questions
56. **"What's the statistical significance of these correlation values?"**
57. **"How reliable are these rule mining results?"**
58. **"What's the sample size for these analyses?"**
59. **"How do the confidence intervals affect interpretation?"**
60. **"What assumptions underlie these statistical measures?"**

### Methodology Questions
61. **"How was the Pareto front calculated for rule mining?"**
62. **"What distance correlation method was used?"**
63. **"How were the chiplet categories (low, medium, high) defined?"**
64. **"What's the difference between this analysis and other data mining approaches?"**
65. **"How robust are these results to different parameter settings?"**

## Usage Instructions

### For Users:
- Ask these questions in natural language
- Be specific about which analysis you're referring to
- Combine questions for deeper insights
- Ask follow-up questions based on initial responses

### For Developers:
- The AI system uses structured data to answer these questions
- Questions are routed to appropriate analysis context
- Responses are tailored to the specific data mining type
- Follow-up questions maintain context from previous analyses

### Example Usage:
```
User: "What's the relationship between GPU count and energy consumption?"
AI: [Provides specific correlation analysis with context]

User: "How should I apply this to optimize my design?"
AI: [Provides actionable design recommendations based on the correlation]

User: "What do the rule mining results say about GPU usage?"
AI: [Provides rule-based insights with confidence levels]
```

This comprehensive question set enables users to explore data mining results thoroughly and extract maximum value from the analyses. 