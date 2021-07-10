
const SCORES: [u32; 64 * 13] = calc_scores();

pub struct PieceSquareTables {
}

impl PieceSquareTables {
    pub fn new(_: &Options) -> Self {
        PieceSquareTables {}
    }

    #[inline]
    pub fn get_packed_score(&self, piece: i8, pos: usize) -> u32 {
        unsafe { *SCORES.get_unchecked((piece + 6) as usize * 64 + pos) }
    }

    pub fn recalculate(&mut self, _: &Options) {}
}

const fn calc_scores() -> [u32; 64 * 13] {
    concat(
        combine(BLACK, P, mirror(PAWN_PST), mirror(EG_PAWN_PST)),
        combine(BLACK, N, mirror(KNIGHT_PST), mirror(EG_KNIGHT_PST)),
        combine(BLACK, B, mirror(BISHOP_PST), mirror(EG_BISHOP_PST)),
        combine(BLACK, R, mirror(ROOK_PST), mirror(EG_ROOK_PST)),
        combine(BLACK, Q, mirror(QUEEN_PST), mirror(EG_QUEEN_PST)),
        combine(BLACK, K, mirror(KING_PST), mirror(EG_KING_PST)),
        combine(WHITE, P, PAWN_PST, EG_PAWN_PST),
        combine(WHITE, N, KNIGHT_PST, EG_KNIGHT_PST),
        combine(WHITE, B, BISHOP_PST, EG_BISHOP_PST),
        combine(WHITE, R, ROOK_PST, EG_ROOK_PST),
        combine(WHITE, Q, QUEEN_PST, EG_QUEEN_PST),
        combine(WHITE, K, KING_PST, EG_KING_PST)
    )
}

const fn concat(
    black_pawns: [u32; 64],
    black_knights: [u32; 64],
    black_bishops: [u32; 64],
    black_rooks: [u32; 64],
    black_queens: [u32; 64],
    black_kings: [u32; 64],
    white_pawns: [u32; 64],
    white_knights: [u32; 64],
    white_bishops: [u32; 64],
    white_rooks: [u32; 64],
    white_queens: [u32; 64],
    white_kings: [u32; 64],
) -> [u32; 64 * 13] {
    let mut all: [u32; 64 * 13] = [0; 64 * 13];

    let mut i = 0;
    while i < 64 {
        all[i] = black_kings[i];
        all[i + 1 * 64] = black_queens[i];
        all[i + 2 * 64] = black_rooks[i];
        all[i + 3 * 64] = black_bishops[i];
        all[i + 4 * 64] = black_knights[i];
        all[i + 5 * 64] = black_pawns[i];

        all[i + 7 * 64] = white_pawns[i];
        all[i + 8 * 64] = white_knights[i];
        all[i + 9 * 64] = white_bishops[i];
        all[i + 10 * 64] = white_rooks[i];
        all[i + 11 * 64] = white_queens[i];
        all[i + 12 * 64] = white_kings[i];
        i += 1;
    }

    all
}

const fn combine(color: Color, piece: i8, scores: [i32; 64], eg_scores: [i32; 64]) -> [u32; 64] {
    let mut combined_scores: [u32; 64] = [0; 64];
    let piece_value = get_piece_value(piece as usize);

    let mut i = 0;
    while i < 64 {
        let score = (scores[i] as i16 + piece_value) * (color as i16);
        let eg_score = (eg_scores[i] as i16 + piece_value) * (color as i16);
        combined_scores[i] = pack_scores(score, eg_score);

        i += 1;
    }

    combined_scores
}

const fn mirror(scores: [i32; 64]) -> [i32; 64] {
    let mut output: [i32; 64] = clone(scores);

    let mut col = 0;
    while col < 8 {

        let mut row = 0;
        while row < 4 {
            let opposite_row = 7 - row;

            let pos = col + row * 8;
            let opposite_pos = col + opposite_row * 8;

            let tmp = output[pos];
            output[pos] = output[opposite_pos];
            output[opposite_pos] = tmp;


            row += 1;
        }

        col += 1;
    }

    output
}

const fn clone(input: [i32; 64]) -> [i32; 64] {
    let mut output: [i32; 64] = [0; 64];

    let mut i = 0;
    while i < 64 {
        output[i] = input[i];
        i += 1;
    }

    output
}

