#***********************************************
#      Filename: __init__.py
#   Description: 大模型格式化字段定义（旅游路书）
#***********************************************

from travel_planner.states.critique import Critique
from travel_planner.states.quality import QualityMetric
from travel_planner.states.eval_result import EvaluationResult
from travel_planner.states.itinerary import AgentInputState, AgentState, TripRequirement, DraftItinerary
from travel_planner.states.scout import ScoutState, ScoutOutputState, Summary
from travel_planner.states.coordinator import CoordinatorState, GatherIntel, ItineraryComplete
