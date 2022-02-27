export NineMensMorrisAction
export GamePhase

struct GamePhase
    game_phase::Int
end

const INIT = GamePhase(0)
const MOVE = GamePhase(1)
const MILL = GamePhase(2)

struct NineMensMorrisAction
    game_phase::GamePhase
    action::Union{CartesianIndex{3}, Tuple{CartesianIndex{3}, CartesianIndex{3}}}
end

action_first_index(action::NineMensMorrisAction) = action.game_phase != MOVE ? action.action : action.action[1]
