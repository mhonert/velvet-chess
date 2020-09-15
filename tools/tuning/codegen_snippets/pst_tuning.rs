
pub struct PieceSquareTables {
    white_scores: [u32; 64 * 7],
    black_scores: [u32; 64 * 7],
}

impl PieceSquareTables {
    pub fn new(options: &Options) -> Self {
        PieceSquareTables {
            white_scores: calc_white_scores(options),
            black_scores: calc_black_scores(options),
        }
    }

    pub fn get_packed_score(&self, piece: i8, pos: usize) -> u32 {
        if piece < 0 {
            return self.black_scores[-piece as usize * 64 + pos];
        }

        self.white_scores[piece as usize * 64 + pos as usize]
    }

    pub fn recalculate(&mut self, options: &Options) {
        self.white_scores.copy_from_slice(&calc_white_scores(options));
        self.black_scores.copy_from_slice(&calc_black_scores(options));
    }
}

fn calc_white_scores(options: &Options) -> [u32; 64 * 7] {
    concat(
        combine(WHITE, P, options.get_pawn_pst(), options.get_eg_pawn_pst()),
        combine(WHITE, N, options.get_knight_pst(), options.get_eg_knight_pst()),
        combine(WHITE, B, options.get_bishop_pst(), options.get_eg_bishop_pst()),
        combine(WHITE, R, options.get_rook_pst(), options.get_eg_rook_pst()),
        combine(WHITE, Q, options.get_queen_pst(), options.get_eg_queen_pst()),
        combine(WHITE, K, options.get_king_pst(), options.get_eg_king_pst())
    )
}

fn calc_black_scores(options: &Options) -> [u32; 64 * 7] {
    concat(
        combine(BLACK, P, mirror(options.get_pawn_pst()), mirror(options.get_eg_pawn_pst())),
        combine(BLACK, N, mirror(options.get_knight_pst()), mirror(options.get_eg_knight_pst())),
        combine(BLACK, B, mirror(options.get_bishop_pst()), mirror(options.get_eg_bishop_pst())),
        combine(BLACK, R, mirror(options.get_rook_pst()), mirror(options.get_eg_rook_pst())),
        combine(BLACK, Q, mirror(options.get_queen_pst()), mirror(options.get_eg_queen_pst())),
        combine(BLACK, K, mirror(options.get_king_pst()), mirror(options.get_eg_king_pst())))
}

fn concat(
    pawns: [u32; 64],
    knights: [u32; 64],
    bishops: [u32; 64],
    rooks: [u32; 64],
    queens: [u32; 64],
    kings: [u32; 64],
) -> [u32; 64 * 7] {
    let mut all: [u32; 64 * 7] = [0; 64 * 7];

    let mut i = 0;
    while i < 64 {
        all[i + 1 * 64] = pawns[i];
        all[i + 2 * 64] = knights[i];
        all[i + 3 * 64] = bishops[i];
        all[i + 4 * 64] = rooks[i];
        all[i + 5 * 64] = queens[i];
        all[i + 6 * 64] = kings[i];

        i += 1;
    }

    all
}

fn combine(color: Color, piece: i8, scores: [i32; 64], eg_scores: [i32; 64]) -> [u32; 64] {
    let mut combined_scores: [u32; 64] = [0; 64];
    let piece_value = PIECE_VALUES[piece as usize];

    let mut i = 0;
    while i < 64 {
        let score = (scores[i] as i16 + piece_value) * (color as i16);
        let eg_score = (eg_scores[i] as i16 + piece_value) * (color as i16);
        combined_scores[i] = pack_scores(score, eg_score);

        i += 1;
    }

    combined_scores
}

fn mirror(scores: [i32; 64]) -> [i32; 64] {
    let mut output: [i32; 64] = scores.clone();

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

