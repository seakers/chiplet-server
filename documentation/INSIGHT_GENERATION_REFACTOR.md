# Insight Generation Pipeline Refactor

## Overview

The insight generation pipeline has been refactored to make insights more goal-aware, design-specific, and structured. The new system integrates optimization objective and trace information dynamically, returns structured JSON data, and provides crisp, actionable insights with follow-up question support.

## Key Improvements

### 1. Goal-Aware Insights
- **Dynamic Objective Integration**: Insights now consider the specific optimization goal (energy, time, or both)
- **Trace-Specific Context**: Analysis includes the trace name and run ID for better context
- **Targeted Recommendations**: Prompts are tailored to the specific optimization objective

### 2. Structured Data Output
- **JSON Response Format**: Both insights and structured data are returned
- **UI-Ready Data**: Structured data can be used for charts, tables, and interactive displays
- **Consistent Schema**: Standardized JSON format across all data mining types

### 3. Improved Prompts
- **Concise Responses**: Focused 2-3 sentence summaries instead of verbose explanations
- **Actionable Content**: Specific recommendations rather than general observations
- **Follow-up Ready**: Designed to encourage deeper exploration through questions

### 4. Follow-up Question System
- **Context-Aware Responses**: AI maintains context from previous analyses
- **Structured Data Integration**: Follow-up questions use the same structured data
- **Comprehensive Q&A**: 65+ predefined question types for thorough exploration

## API Changes

### Distance Correlation Insights

**Endpoint**: `GET /api/distance-correlation-insights/`

**New Parameters**:
- `objective`: "energy", "time", or "both" (default: "both")
- `trace_name`: Name of the trace used (default: "Unknown")
- `run_id`: Current run ID for context (optional)

**Response Format**:
```json
{
  "insights": "Concise natural language summary...",
  "structured_data": {
    "high_impact_on_energy": [
      {"chiplet": "GPU", "correlation": 0.81},
      {"chiplet": "Sparse", "correlation": 0.67}
    ],
    "high_impact_on_time": [
      {"chiplet": "Conv", "correlation": 0.75},
      {"chiplet": "Attn", "correlation": 0.60}
    ],
    "trace_name": "gpt-2",
    "objective": "energy",
    "run_id": "run_123"
  }
}
```

### Rule Mining Insights

**Endpoint**: `GET /api/rule-mining-insights/`

**New Parameters**:
- `objective`: "energy", "time", or "both" (default: "both")
- `trace_name`: Name of the trace used (default: "Unknown")
- `run_id`: Current run ID for context (optional)
- Plus existing rule mining parameters

**Response Format**:
```json
{
  "insights": "Concise natural language summary...",
  "structured_data": {
    "rules": [
      {
        "conditions": ["High GPU", "Low Attn"],
        "confidence_f_to_p": 0.85,
        "confidence_p_to_f": 0.72,
        "lift": 2.3
      }
    ],
    "trace_name": "gpt-2",
    "objective": "energy",
    "run_id": "run_123",
    "analysis_region": "pareto (ranks 1-3)"
  }
}
```

### Follow-up Questions

**Endpoint**: `POST /api/data-mining-followup/`

**Request Format**:
```json
{
  "question": "What's the relationship between GPU count and energy consumption?",
  "data_mining_type": "distance_correlation",
  "structured_data": { /* structured data from previous analysis */ }
}
```

**Response Format**:
```json
{
  "response": "Detailed answer based on the structured data..."
}
```

## Frontend Integration

### Component Updates

**DistanceCorrelation.vue**:
- Passes optimization context parameters
- Stores structured data for follow-up questions
- Emits insights to chat system

**RuleMining.vue**:
- Passes optimization context parameters
- Includes rule mining specific parameters
- Stores structured data for follow-up questions

### Service Updates

**analytics.js**:
- Updated function signatures to accept parameters
- New `askDataMiningFollowup()` function
- Enhanced documentation for all functions

## Usage Examples

### Basic Insight Generation

```javascript
// Get distance correlation insights with context
const params = {
  objective: 'energy',
  trace_name: 'gpt-2',
  run_id: 'run_123'
};

const response = await getDistanceCorrelationInsights(params);
console.log(response.insights); // Natural language summary
console.log(response.structured_data); // JSON data for UI
```

### Follow-up Questions

```javascript
// Ask follow-up question about distance correlation
const followupData = {
  question: "What's the optimal GPU count for energy efficiency?",
  data_mining_type: "distance_correlation",
  structured_data: response.structured_data
};

const followupResponse = await askDataMiningFollowup(followupData);
console.log(followupResponse.response); // Detailed answer
```

### Rule Mining with Context

```javascript
// Get rule mining insights with optimization context
const ruleParams = {
  objective: 'time',
  trace_name: 'bert-base',
  run_id: 'run_456',
  region: 'pareto',
  paretoStartRank: 1,
  paretoEndRank: 5
};

const ruleResponse = await getRuleMiningInsights(ruleParams);
console.log(ruleResponse.insights); // Rule-based summary
console.log(ruleResponse.structured_data.rules); // Parsed rules
```

## Prompt Engineering

### Distance Correlation Prompts

**Goal-Specific Prompts**:
- Energy focus: "The current design goal is to minimize energy consumption"
- Time focus: "The current design goal is to minimize execution time"
- Both: "The current design goal is to optimize both energy and execution time"

**Structured Format**:
- JSON data inclusion for context
- Specific question categories
- Concise response requirements

### Rule Mining Prompts

**Analysis Context**:
- Pareto-optimal point focus
- Trace-specific information
- Rule confidence and lift explanation

**Actionable Output**:
- Pattern identification
- Specific recommendations
- Conflict detection

## Follow-up Question Categories

### Distance Correlation (20 questions)
- High-level analysis
- Specific correlations
- Design decisions
- Comparative analysis

### Rule Mining (25 questions)
- Rule patterns
- Confidence analysis
- Specific rules
- Design application
- Comparative rules

### Cross-Analysis (10 questions)
- Integration insights
- Practical application

### Technical Deep-Dive (10 questions)
- Statistical significance
- Methodology details

## Benefits

### For Users
- **Faster Insights**: Concise, actionable summaries
- **Deeper Exploration**: Comprehensive follow-up question system
- **Context-Aware**: Optimization goal and trace-specific insights
- **Interactive**: Natural language Q&A with structured data

### For Developers
- **Structured Data**: JSON format for UI integration
- **Extensible**: Easy to add new question types
- **Maintainable**: Clear separation of concerns
- **Testable**: Structured input/output format

### For System Performance
- **Reduced Token Usage**: Concise prompts and responses
- **Better Context**: Relevant information only
- **Faster Responses**: Focused analysis
- **Scalable**: Modular design for new features

## Migration Guide

### Backend Changes
1. Updated `distance_correlation_insights()` function
2. Updated `rule_mining_insights()` function
3. Added `data_mining_followup()` endpoint
4. Added `get_data_mining_followup_response()` method to ChatBotModel
5. Updated URL routing

### Frontend Changes
1. Updated analytics service functions
2. Modified component parameter passing
3. Added structured data storage
4. Enhanced error handling

### Database Changes
- No database changes required
- All data is passed through API parameters

## Future Enhancements

### Planned Features
1. **Multi-Analysis Integration**: Combine insights from multiple data mining types
2. **Historical Comparison**: Compare insights across different runs
3. **Automated Recommendations**: Generate design suggestions based on insights
4. **Interactive Visualizations**: Use structured data for dynamic charts
5. **Export Functionality**: Save insights and structured data for external use

### Extensibility
- Easy to add new data mining types
- Modular prompt system
- Pluggable follow-up question handlers
- Configurable response formats

This refactored pipeline provides a more intelligent, context-aware, and user-friendly insight generation system that maximizes the value of data mining results for chiplet design optimization. 