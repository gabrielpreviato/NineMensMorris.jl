export NineMensMorrisEnv

using ReinforcementLearning

using Base: show

NineMensMorrisObs = Observation{Tuple{BitArray{4}, GamePhase}}

Base.@kwdef mutable struct NineMensMorrisEnv <: AbstractEnv
    player::Player
    orig_player::Player
    board::BitArray{4}
    game_phase::GamePhase=INIT
    done::Bool=false
    winner::Union{Nothing, Player}
end

function Base.show(io::IO, env::NineMensMorrisEnv)
    function get_letter(board, index)
        index = CartesianIndex(index)
        if board[index, 1]
            "O"
        elseif board[index, 2]
            "W"
        elseif board[index, 3]
            "B"
        else
            "?"
        end
    end
    
    str = 
    "$(get_letter(env.board, (1, 1, 1)))-----$(get_letter(env.board, (1, 2, 1)))-----$(get_letter(env.board, (1, 3, 1)))    |" * "\n" *
    "| |  |  |   | |" * "\n" *
    "| $(get_letter(env.board, (1, 1, 2)))---$(get_letter(env.board, (1, 2, 2)))---$(get_letter(env.board, (1, 3, 2)))    |" * "\n" *
    "| |  |  |   | |" * "\n" *
    "|    $(get_letter(env.board, (1, 1, 3)))-$(get_letter(env.board, (1, 2, 3)))-$(get_letter(env.board, (1, 3, 3)))  | |" * "\n" *
    "| |  |  |   | |" * "\n" *
    "$(get_letter(env.board, (2, 1, 1)))-$(get_letter(env.board, (2, 1, 2)))-$(get_letter(env.board, (2, 1, 3)))-*-$(get_letter(env.board, (2, 3, 3)))-$(get_letter(env.board, (2, 3, 2)))-$(get_letter(env.board, (2, 3, 1)))" * "\n" *
    "| |  |  |   | |" * "\n" *
    "|    $(get_letter(env.board, (3, 1, 3)))-$(get_letter(env.board, (3, 2, 3)))-$(get_letter(env.board, (3, 3, 3)))  | |" * "\n" *
    "| |  |  |   | |" * "\n" *
    "| $(get_letter(env.board, (3, 1, 2)))---$(get_letter(env.board, (3, 2, 2)))---$(get_letter(env.board, (3, 3, 2)))  | |" * "\n" *
    "| |  |  |   | |" * "\n" *
    "$(get_letter(env.board, (3, 1, 1)))-----$(get_letter(env.board, (3, 2, 1)))-----$(get_letter(env.board, (3, 3, 1)))    |"

    print(io, str)
end

function NineMensMorrisEnv(player::Player)
    board = BitArray{4}(undef, 3, 3, 3, 3)
    reset_board!(board)
    
    NineMensMorrisEnv(player, player, board, INIT, false, nothing)
end

function NineMensMorrisEnv(player_char::String)
    if player_char == "W"
        NineMensMorrisEnv(WHITES)
    elseif player_char == "B"
        NineMensMorrisEnv(BLACKS)
    end
end

function NineMensMorrisEnv() 
    NineMensMorrisEnv(WHITES)
end

function reset_board!(env::NineMensMorrisEnv)
    reset_board!(env.board)
end

function reset_board!(board::BitArray{4})
    fill!(board, false)
    
    board[:, :, :, 1] .= true
    board[2, 2, :, 1] .= false
end

function (env::NineMensMorrisEnv)(action::NineMensMorrisAction)
    # if env.game_phase != INIT
    #     println(action)
    #     println(env.player)
    #     println(legal_action_space(env))
    # end

    _exec_action(env, action)
    _env_step(env)
    _check_stop(env)

    # println("In action!")

    # println(env)

    # println(legal_action_space_mask(env))
    # println(count(legal_action_space_mask(env)))
end

# RLBase.state(env::NineMensMorrisEnv) = reshape(env.board * 2^env.game_phase.game_phase, (81))
RLBase.state(env::NineMensMorrisEnv) = reshape(env.board * 2^env.game_phase.game_phase, (9, 3, 3))

RLBase.state_space(::NineMensMorrisEnv) =
    Space(fill([false, true], 9, 3, 3))

init_action_space(::NineMensMorrisEnv) = [NineMensMorrisAction(INIT, i) for i in setdiff(vec(CartesianIndices((1:3, 1:3, 1:3))), vec(CartesianIndices((2:2, 2:2, 1:3))))]
init_action_space(::NineMensMorrisEnv, p::Player) = [NineMensMorrisAction(INIT, i) for i in setdiff(vec(CartesianIndices((1:3, 1:3, p.board_i))), vec(CartesianIndices((2:2, 2:2, p.board_i))))]

mill_action_space(::NineMensMorrisEnv) = [NineMensMorrisAction(MILL, i) for i in setdiff(vec(CartesianIndices((1:3, 1:3, 1:3))), vec(CartesianIndices((2:2, 2:2, 1:3))))]
mill_action_space(::NineMensMorrisEnv, p::Player) = [NineMensMorrisAction(MILL, i) for i in setdiff(vec(CartesianIndices((1:3, 1:3, (!p).board_i))), vec(CartesianIndices((2:2, 2:2, (!p).board_i))))]

function move_action_space(::NineMensMorrisEnv, board_range::UnitRange{Int})
    move_actions = Array{NineMensMorrisAction, 1}()
    cartesian_indices = Array{CartesianIndex, 1}()
    for i in 1:3
        for j in 1:3
            for k in board_range
                i != 2 && j != 2 ? push!(cartesian_indices,  CartesianIndex(i, j, k)) : nothing
           end
        end
    end

    for i in 1:3
        for j in 1:3
            for k in board_range
                for base_index in cartesian_indices
                   (i != 2 || j != 2) && (i != base_index[1] || j != base_index[2] || k != base_index[3]) ? push!(move_actions, NineMensMorrisAction(MOVE, (base_index, CartesianIndex(i, j, k)))) : nothing
                end
            end
        end
    end

    return move_actions
end

move_action_space(env::NineMensMorrisEnv, p::Player) = move_action_space(env, p.board_i)
move_action_space(env::NineMensMorrisEnv) = move_action_space(env, 1:3)

function RLBase.action_space(env::NineMensMorrisEnv)
    init_actions = init_action_space(env)
    mill_actions = mill_action_space(env)
    move_actions = move_action_space(env)

    return [init_actions; mill_actions; move_actions]
end

RLBase.legal_action_space(env::NineMensMorrisEnv) = legal_action_space(env, env.player)
RLBase.legal_action_space_mask(env::NineMensMorrisEnv) = legal_action_space_mask(env, env.player)

RLBase.legal_action_space(env::NineMensMorrisEnv, p::Player) = action_space(env)[legal_action_space_mask(env, p)]

function RLBase.legal_action_space_mask(env::NineMensMorrisEnv, p::Player)
	legal_actions = Array{NineMensMorrisAction, 1}()
    total_actions::Array{NineMensMorrisAction, 1} = action_space(env)

	if env.game_phase == INIT
		legal_actions = [NineMensMorrisAction(INIT, action.action) for action in init_action_space(env, p) if env.board[action_first_index(action), 1]]

	elseif env.game_phase == MILL
		legal_actions = [NineMensMorrisAction(MILL, action.action) for action in mill_action_space(env, p) if env.board[action_first_index(action), (!p).board_i]]
		
	elseif env.game_phase == MOVE
		empty_spaces = env.board[:, :, :, 1]
		player_spaces = env[p]
	
        if length(findall(env[p])) > 3
            for i in 1:3
                # Down moves
                player_spaces[1, 1, i] && empty_spaces[2, 1, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(1, 1, i), CartesianIndex(2, 1, i))))
                player_spaces[2, 1, i] && empty_spaces[3, 1, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 1, i), CartesianIndex(3, 1, i))))
        
                player_spaces[1, 3, i] && empty_spaces[2, 3, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(1, 3, i), CartesianIndex(2, 3, i))))
                player_spaces[2, 3, i] && empty_spaces[3, 3, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 3, i), CartesianIndex(3, 3, i))))
        
                # Up moves
                player_spaces[2, 1, i] && empty_spaces[1, 1, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 1, i), CartesianIndex(1, 1, i))))
                player_spaces[3, 1, i] && empty_spaces[2, 1, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(3, 1, i), CartesianIndex(2, 1, i))))
        
                player_spaces[2, 3, i] && empty_spaces[1, 3, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 3, i), CartesianIndex(1, 3, i))))
                player_spaces[3, 3, i] && empty_spaces[2, 3, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(3, 3, i), CartesianIndex(2, 3, i))))
        
                # Rigth moves
                player_spaces[1, 1, i] && empty_spaces[1, 2, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(1, 1, i), CartesianIndex(1, 2, i))))
                player_spaces[1, 2, i] && empty_spaces[1, 3, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(1, 2, i), CartesianIndex(1, 3, i))))
        
                player_spaces[3, 1, i] && empty_spaces[3, 2, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(3, 1, i), CartesianIndex(3, 2, i))))
                player_spaces[3, 2, i] && empty_spaces[3, 1, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(3, 2, i), CartesianIndex(3, 1, i))))
        
                # Left moves
                player_spaces[1, 2, i] && empty_spaces[1, 1, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(1, 2, i), CartesianIndex(1, 1, i))))
                player_spaces[1, 3, i] && empty_spaces[1, 2, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(1, 3, i), CartesianIndex(1, 2, i))))
        
                player_spaces[3, 2, i] && empty_spaces[3, 1, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(3, 2, i), CartesianIndex(3, 1, i))))
                player_spaces[3, 3, i] && empty_spaces[3, 2, i] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(3, 3, i), CartesianIndex(3, 2, i))))
            end
                
            # Left-Right transversal move
            player_spaces[2, 1, 1] && empty_spaces[2, 1, 2] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 1, 1), CartesianIndex(2, 1, 2))))
            player_spaces[2, 1, 2] && empty_spaces[2, 1, 3] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 1, 2), CartesianIndex(2, 1, 3))))
            
            player_spaces[2, 1, 3] && empty_spaces[2, 1, 2] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 1, 3), CartesianIndex(2, 1, 2))))
            player_spaces[2, 1, 2] && empty_spaces[2, 1, 1] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 1, 2), CartesianIndex(2, 1, 1))))
            
            # Left-Right transversal move
            player_spaces[2, 3, 1] && empty_spaces[2, 3, 2] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 3, 1), CartesianIndex(2, 3, 2))))
            player_spaces[2, 3, 2] && empty_spaces[2, 3, 3] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 3, 2), CartesianIndex(2, 3, 3))))
            
            player_spaces[2, 3, 3] && empty_spaces[2, 3, 2] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 3, 3), CartesianIndex(2, 3, 2))))
            player_spaces[2, 3, 2] && empty_spaces[2, 3, 1] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 3, 2), CartesianIndex(2, 3, 1))))
            
            # Top-Down transversal move
            player_spaces[1, 2, 1] && empty_spaces[2, 1, 2] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 1, 1), CartesianIndex(2, 1, 2))))
            player_spaces[1, 2, 2] && empty_spaces[2, 1, 3] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 1, 2), CartesianIndex(2, 1, 3))))
            
            player_spaces[1, 2, 3] && empty_spaces[2, 1, 2] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 1, 3), CartesianIndex(2, 1, 2))))
            player_spaces[1, 2, 2] && empty_spaces[2, 1, 1] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 1, 2), CartesianIndex(2, 1, 1))))
            
            # Down-Top transversal move
            player_spaces[2, 3, 1] && empty_spaces[2, 3, 2] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 3, 1), CartesianIndex(2, 3, 2))))
            player_spaces[2, 3, 2] && empty_spaces[2, 3, 3] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 3, 2), CartesianIndex(2, 3, 3))))
            
            player_spaces[2, 3, 3] && empty_spaces[2, 3, 2] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 3, 3), CartesianIndex(2, 3, 2))))
            player_spaces[2, 3, 2] && empty_spaces[2, 3, 1] && push!(legal_actions, NineMensMorrisAction(MOVE, (CartesianIndex(2, 3, 2), CartesianIndex(2, 3, 1))))
        else
            player_pieces = findall(env[p])
            empty_pieces = findall(empty_spaces)
            legal_actions = [NineMensMorrisAction(MOVE, (x, y)) for x in player_pieces, y in empty_pieces]
            
        end       
	end

    mask_indices = findall(x -> in(x, legal_actions), total_actions)
    actions_mask = fill(false, length(total_actions))
    actions_mask[mask_indices] .= true

    return actions_mask
end

function _check_stop(env)
	if env.game_phase != INIT && sum(env[WHITES]) <= 2
		env.done = true
		env.winner = BLACKS
	elseif env.game_phase != INIT && sum(env[BLACKS]) <= 2
		env.done = true
		env.winner = WHITES
	elseif length(legal_action_space(env, env.player)) == 0
		env.done = true
		env.winner = env.player
        # println("No more moves for opponent!")
	end
end

function _exec_action(env::NineMensMorrisEnv, action::NineMensMorrisAction)
    if action ∉ legal_action_space(env)
        @error "Action {$(action.game_phase)} not legal {$(legal_action_space(env))}"
        return
    end
    
	if env.game_phase != action.game_phase 
		@error "Action phase {$(action.game_phase)} is different than expected by Environment {$(env.game_phase)}"
        @error "$env"
        @error "$action"
        # println(legal_action_space(env))
	end

	game_phase = action.game_phase
	if game_phase == INIT
		placement = action.action
		env.board[placement, 1] = false
	    env.board[placement, env.player.board_i] = true    
	elseif game_phase == MOVE
		old_placement = action.action[1]
		new_placement = action.action[2]
		
		env.board[old_placement, 1] = true
		env.board[old_placement, env.player.board_i] = false

		env.board[new_placement, 1] = false
	    env.board[new_placement, env.player.board_i] = true
	elseif game_phase == MILL
		placement = action.action
		env.board[placement, 1] = true
	    env.board[placement, (!env.player).board_i] = false
	end
end

function _is_mill_possible(env::NineMensMorrisEnv)
		# Check top rows
		any(mapslices(all, env[env.player][1, :, :], dims=1)) ||
	
		# Check bottom rows
		any(mapslices(all, env[env.player][3, :, :], dims=1)) ||
	
		# Check left cols
		any(mapslices(all, env[env.player][:, 1, :], dims=1)) ||
	
		# Check right cols
		any(mapslices(all, env[env.player][:, 3, :], dims=1)) ||
	
		# Check left inter
		all(env[env.player][2, 1, :]) ||
	
		# Check rigth inter
		all(env[env.player][2, 3, :]) ||
	
		# Check top inter
		all(env[env.player][1, 2, :]) ||
	
		# Check bottom inter
		all(env[env.player][3, 2, :])
	
end

function _env_step(env::NineMensMorrisEnv)
	if env.game_phase == INIT
        # println("board trues:", count(env.board[:, :, :, 1]))
        # println("whites trues:", count(env.board[:, :, :, 2]))
        # println("blacks trues:", count(env.board[:, :, :, 3]))
		if count(env.board[:, :, :, 1]) == 6 && count(env.board[:, :, :, 2]) == 9 && count(env.board[:, :, :, 3]) == 9
            # println("End of INIT")
			if _is_mill_possible(env)
				env.game_phase = MILL
                # println("Changing INIT to MILL")
                return
			else
				env.game_phase = MOVE
			end
		end
	elseif env.game_phase == MOVE
		if _is_mill_possible(env)
			env.game_phase = MILL
            return
		end
	elseif env.game_phase == MILL
		env.game_phase = MOVE
	end

    env.player = !env.player
end

RLBase.is_terminated(env::NineMensMorrisEnv) = env.done
Base.hash(env::NineMensMorrisEnv, h::UInt) = hash(env.board, h)
Base.isequal(a::NineMensMorrisEnv, b::NineMensMorrisEnv) = isequal(a.board, b.board)
Base.getindex(env::NineMensMorrisEnv, p::Player) = env.board[:, :, :, p.board_i]
RLBase.current_player(env::NineMensMorrisEnv) = env.player
RLBase.players(::NineMensMorrisEnv) = (WHITES, BLACKS)
opp_player(env::NineMensMorrisEnv) = !env.player

function RLBase.reset!(env::NineMensMorrisEnv)
	reset_board!(env)
	env.game_phase = INIT
	env.done = false
    env.winner = nothing
	
    env.player = env.orig_player
    # env.orig_player = env.player

    # println("reset")
end

function RLBase.reward(env::NineMensMorrisEnv)
    if env.player == env.orig_player
        if is_terminated(env)
            winner = env.winner
            if isnothing(winner)
                -50
            elseif winner === env.player
                250
            else
                -250
            end
        else
            if env.game_phase == MOVE
                count(env[env.player]) + 10*(9 - count(env[!env.player]))
            elseif env.game_phase == MILL
                10
            else
                0
            end
        end
    else
        0
    end
end

function is_win(env::NineMensMorrisEnv, player::Player)
	opp_player::Player = setdiff(players(env), [current_player(env)])[1]
	if length(legal_action_space(env, player)) == 0
		return nothing
	elseif length(findall(env[opp_player])) <= 2
		return true
	end

	return false
end

RLBase.NumAgentStyle(::NineMensMorrisEnv) = MultiAgent(2)
RLBase.DynamicStyle(::NineMensMorrisEnv) = SEQUENTIAL
RLBase.ActionStyle(::NineMensMorrisEnv) = FULL_ACTION_SET
RLBase.InformationStyle(::NineMensMorrisEnv) = PERFECT_INFORMATION
RLBase.StateStyle(::NineMensMorrisEnv) = NineMensMorrisObs()
RLBase.RewardStyle(::NineMensMorrisEnv) = STEP_REWARD
RLBase.UtilityStyle(::NineMensMorrisEnv) = ZERO_SUM
RLBase.ChanceStyle(::NineMensMorrisEnv) = DETERMINISTIC


using StableRNGs
using Flux
using Flux.Losses
using BSON

# Dense(256, 512, relu; init=glorot_uniform(rng)),
#                         Dense(512, 512, relu; init=glorot_uniform(rng)),
#                         Dropout(0.1),
#                         Dense(512, na; init = glorot_uniform(rng)),
#                     ),
#                     critic = Chain(
#                         Dense(ns, 256, relu; init = glorot_uniform(rng)),
#                         Dense(256, 512, relu; init=glorot_uniform(rng)),
#                         Dense(512, 512, relu; init=glorot_uniform(rng)),
#                         Dropout(0.1),
#                         Dense(512, 1; init = glorot_uniform(rng)),

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:NineMensMorrisEnv},
    ::Nothing;
    seed = 123
)
    rng = StableRNG(seed)
    N_ENV = 1
    UPDATE_FREQ = 10
    env = MultiThreadEnv([
        discrete2standard_discrete(NineMensMorrisEnv("W")) for i in 1:N_ENV
    ])
    
    ns, na = size(state(env[1])), length(action_space(env[1]))
    println(ns)
    nf = length(state(env[1]))

    agent = Agent(
        policy = QBasedPolicy(
            learner = A2CLearner(
                approximator = ActorCritic(
                    actor = Chain(
                        Conv((3, 3), ns[3] => 16, relu; init=glorot_uniform(rng), pad=2),
                        Conv((3, 3), 16 => 32, relu; init=glorot_uniform(rng), pad=2),
                        Conv((3, 3), 32 => 64, relu; init=glorot_uniform(rng), pad=2),
                        MaxPool((2, 2)),
                        Conv((3, 3), 64 => 8, relu; init=glorot_uniform(rng), pad=2),
                        Flux.flatten,
                        Dense(432, na; init=glorot_uniform(rng)),
                        # Dense(na*2, na; init=glorot_uniform(rng)),
                    ),
                    critic = Chain(
                        Conv((3, 3), ns[3] => 16, relu; init=glorot_uniform(rng), pad=2),
                        Conv((3, 3), 16 => 32, relu; init=glorot_uniform(rng), pad=2),
                        Conv((3, 3), 32 => 64, relu; init=glorot_uniform(rng), pad=2),
                        MaxPool((2, 2)),
                        Conv((3, 3), 64 => 8, relu; init=glorot_uniform(rng), pad=2),
                        Flux.flatten,
                        Dense(432, na; init=glorot_uniform(rng)),
                        # Dense(na*2, na; init=glorot_uniform(rng)),
                    ),
                    optimizer = ADAM(1e-3),
                ) |> gpu,
                γ = 0.99f0,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.001f0,
                update_freq = UPDATE_FREQ,
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer()),
        ),
        trajectory = CircularArraySARTTrajectory(;
            capacity = UPDATE_FREQ,
            state = Matrix{Float32} => (nf, N_ENV),
            action = Vector{Int} => (N_ENV,),
            reward = Vector{Int} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )

    parameters_dir = mktempdir()

    stop_condition = StopAfterStep(1_000_000, is_show_progress=true)
    hook = ComposedHook(
        TotalBatchRewardPerEpisode(N_ENV, is_display_on_exit=true),
        DoEveryNStep(n=50_000) do t, p, e
            ps = params(p)
            f = joinpath(parameters_dir, "parameters_at_step_$t.bson")
            BSON.@save f ps
            println("parameters at step $t saved to $f")
        end
        )
    
    Experiment(agent, env, stop_condition, hook, "# A2C <-> NineMensMorris")
end
