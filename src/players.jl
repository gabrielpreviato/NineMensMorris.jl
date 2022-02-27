import Base: !

struct Player
    board_i::Int
end

const WHITES = Player(2)
const BLACKS = Player(3)

Base.:(!)(p::Player) = p == WHITES ? BLACKS : WHITES
