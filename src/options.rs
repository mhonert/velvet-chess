/*
 * Velvet Chess Engine
 * Copyright (C) 2020 mhonert (https://github.com/mhonert)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// Auto-generated file (see tools/tuning/gencode.py)

use std::cmp::max;
use std::sync::mpsc::Sender;
use crate::engine::Message;
use std::str::FromStr;
use crate::colors::{Color, BLACK, WHITE};
use crate::pieces::{B, K, N, P, PIECE_VALUES, Q, R};
use crate::score_util::{pack_scores};

pub struct Options {
    castling_bonus: i32,
    lost_queenside_castling_penalty: i32,
    lost_kingside_castling_penalty: i32,
    doubled_pawn_penalty: i32,
    king_shield_bonus: i32,
    pawn_cover_bonus: i32,
    knight_king_threat: i32,
    bishop_king_threat: i32,
    rook_king_threat: i32,
    queen_king_threat: i32,
    eg_passed_pawn_bonus: [i32; 4],
    passed_pawn_bonus: [i32; 4],
    passed_pawn_king_defense_bonus: [i32; 8],
    passed_pawn_king_attacked_penalty: [i32; 8],
    king_danger_piece_penalty: [i32; 16],
    king_threat_adjustment: [i32; 128],
    eg_knight_mob_bonus: [i32; 9],
    eg_bishop_mob_bonus: [i32; 14],
    eg_rook_mob_bonus: [i32; 15],
    eg_queen_mob_bonus: [i32; 28],
    knight_mob_bonus: [i32; 9],
    bishop_mob_bonus: [i32; 14],
    rook_mob_bonus: [i32; 15],
    queen_mob_bonus: [i32; 28],
    eg_pawn_pst: [i32; 64],
    pawn_pst: [i32; 64],
    eg_knight_pst: [i32; 64],
    knight_pst: [i32; 64],
    eg_bishop_pst: [i32; 64],
    bishop_pst: [i32; 64],
    eg_rook_pst: [i32; 64],
    rook_pst: [i32; 64],
    eg_queen_pst: [i32; 64],
    queen_pst: [i32; 64],
    eg_king_pst: [i32; 64],
    king_pst: [i32; 64],
}

impl Options {
    pub fn new() -> Self {
        Options{
            castling_bonus: 28,
            lost_queenside_castling_penalty: 24,
            lost_kingside_castling_penalty: 51,
            doubled_pawn_penalty: 19,
            king_shield_bonus: 12,
            pawn_cover_bonus: 12,
            knight_king_threat: 1,
            bishop_king_threat: 1,
            rook_king_threat: 2,
            queen_king_threat: 2,
            eg_passed_pawn_bonus: [90, 91, 58, 0],
            passed_pawn_bonus: [294, 109, 0, 0],
            passed_pawn_king_defense_bonus: [0, 72, 55, 22, 5, 0, 0, 0],
            passed_pawn_king_attacked_penalty: [0, 121, 68, 42, 24, 6, 0, 1],
            king_danger_piece_penalty: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            king_threat_adjustment: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            eg_knight_mob_bonus: [-282, -207, -154, -120, -89, -61, -53, -54, -82],
            eg_bishop_mob_bonus: [-168, -115, -90, -66, -53, -51, -47, -49, -44, -42, -52, -76, -87, -120],
            eg_rook_mob_bonus: [-179, -135, -74, -74, -53, -46, -36, -38, -30, -34, -35, -48, -53, -66, -135],
            eg_queen_mob_bonus: [-311, -205, -156, -93, -44, -37, -23, -9, -3, 1, 12, -3, -21, -50, -69, -110, -123, -149, -175, -192, -173, -133, -173, -166, -154, -86, 24, 93],
            knight_mob_bonus: [-33, -12, -3, 5, 2, -5, -8, 0, 9],
            bishop_mob_bonus: [14, 6, 15, 21, 26, 36, 37, 41, 43, 46, 66, 119, 125, 255],
            rook_mob_bonus: [-22, -16, -42, -27, -27, -16, -15, -3, 2, 16, 24, 51, 66, 99, 236],
            queen_mob_bonus: [0, 3, 0, -3, -4, 1, 2, 1, 3, 6, 8, 21, 43, 75, 105, 168, 188, 231, 277, 311, 279, 182, 296, 241, 239, -46, -117, -331],
            eg_pawn_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pawn_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            eg_knight_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            knight_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            eg_bishop_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bishop_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            eg_rook_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            rook_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            eg_queen_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            queen_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            eg_king_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            king_pst: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    }
    
    pub fn set_option(&mut self, name: String, value: i32) {
        match name.as_str() {
            "castlingbonus" => self.set_castling_bonus(value),
            "lostqueensidecastlingpenalty" => self.set_lost_queenside_castling_penalty(value),
            "lostkingsidecastlingpenalty" => self.set_lost_kingside_castling_penalty(value),
            "doubledpawnpenalty" => self.set_doubled_pawn_penalty(value),
            "kingshieldbonus" => self.set_king_shield_bonus(value),
            "pawncoverbonus" => self.set_pawn_cover_bonus(value),
            "knightkingthreat" => self.set_knight_king_threat(value),
            "bishopkingthreat" => self.set_bishop_king_threat(value),
            "rookkingthreat" => self.set_rook_king_threat(value),
            "queenkingthreat" => self.set_queen_king_threat(value),
            _ => println!("Unknown option {}", name)
        }
    }
    
    pub fn set_array_option(&mut self, name: String, index: usize, value: i32) {
        match name.as_str() {
            "egpassedpawnbonus" => self.set_eg_passed_pawn_bonus(index, value),
            "passedpawnbonus" => self.set_passed_pawn_bonus(index, value),
            "passedpawnkingdefensebonus" => self.set_passed_pawn_king_defense_bonus(index, value),
            "passedpawnkingattackedpenalty" => self.set_passed_pawn_king_attacked_penalty(index, value),
            "kingdangerpiecepenalty" => self.set_king_danger_piece_penalty(index, value),
            "kingthreatadjustment" => self.set_king_threat_adjustment(index, value),
            "egknightmobbonus" => self.set_eg_knight_mob_bonus(index, value),
            "egbishopmobbonus" => self.set_eg_bishop_mob_bonus(index, value),
            "egrookmobbonus" => self.set_eg_rook_mob_bonus(index, value),
            "egqueenmobbonus" => self.set_eg_queen_mob_bonus(index, value),
            "knightmobbonus" => self.set_knight_mob_bonus(index, value),
            "bishopmobbonus" => self.set_bishop_mob_bonus(index, value),
            "rookmobbonus" => self.set_rook_mob_bonus(index, value),
            "queenmobbonus" => self.set_queen_mob_bonus(index, value),
            "egpawnpst" => self.set_eg_pawn_pst(index, value),
            "pawnpst" => self.set_pawn_pst(index, value),
            "egknightpst" => self.set_eg_knight_pst(index, value),
            "knightpst" => self.set_knight_pst(index, value),
            "egbishoppst" => self.set_eg_bishop_pst(index, value),
            "bishoppst" => self.set_bishop_pst(index, value),
            "egrookpst" => self.set_eg_rook_pst(index, value),
            "rookpst" => self.set_rook_pst(index, value),
            "egqueenpst" => self.set_eg_queen_pst(index, value),
            "queenpst" => self.set_queen_pst(index, value),
            "egkingpst" => self.set_eg_king_pst(index, value),
            "kingpst" => self.set_king_pst(index, value),
            _ => println!("Unknown option {}", name)
        }
    }
    
    fn set_castling_bonus(&mut self, value: i32) {
        self.castling_bonus = value;
    }

    #[inline]
    pub fn get_castling_bonus(&self) -> i32 {
        self.castling_bonus
    }

    fn set_lost_queenside_castling_penalty(&mut self, value: i32) {
        self.lost_queenside_castling_penalty = value;
    }

    #[inline]
    pub fn get_lost_queenside_castling_penalty(&self) -> i32 {
        self.lost_queenside_castling_penalty
    }

    fn set_lost_kingside_castling_penalty(&mut self, value: i32) {
        self.lost_kingside_castling_penalty = value;
    }

    #[inline]
    pub fn get_lost_kingside_castling_penalty(&self) -> i32 {
        self.lost_kingside_castling_penalty
    }

    fn set_doubled_pawn_penalty(&mut self, value: i32) {
        self.doubled_pawn_penalty = value;
    }

    #[inline]
    pub fn get_doubled_pawn_penalty(&self) -> i32 {
        self.doubled_pawn_penalty
    }

    fn set_king_shield_bonus(&mut self, value: i32) {
        self.king_shield_bonus = value;
    }

    #[inline]
    pub fn get_king_shield_bonus(&self) -> i32 {
        self.king_shield_bonus
    }

    fn set_pawn_cover_bonus(&mut self, value: i32) {
        self.pawn_cover_bonus = value;
    }

    #[inline]
    pub fn get_pawn_cover_bonus(&self) -> i32 {
        self.pawn_cover_bonus
    }

    fn set_knight_king_threat(&mut self, value: i32) {
        self.knight_king_threat = value;
    }

    #[inline]
    pub fn get_knight_king_threat(&self) -> i32 {
        self.knight_king_threat
    }

    fn set_bishop_king_threat(&mut self, value: i32) {
        self.bishop_king_threat = value;
    }

    #[inline]
    pub fn get_bishop_king_threat(&self) -> i32 {
        self.bishop_king_threat
    }

    fn set_rook_king_threat(&mut self, value: i32) {
        self.rook_king_threat = value;
    }

    #[inline]
    pub fn get_rook_king_threat(&self) -> i32 {
        self.rook_king_threat
    }

    fn set_queen_king_threat(&mut self, value: i32) {
        self.queen_king_threat = value;
    }

    #[inline]
    pub fn get_queen_king_threat(&self) -> i32 {
        self.queen_king_threat
    }

    fn set_eg_passed_pawn_bonus(&mut self, index: usize, value: i32) {
        self.eg_passed_pawn_bonus[index] = max(0, value);
    }
                
    #[inline]
    pub fn get_eg_passed_pawn_bonus(&self, index: usize) -> i32 {
        self.eg_passed_pawn_bonus[index]
    }
    
    fn set_passed_pawn_bonus(&mut self, index: usize, value: i32) {
        self.passed_pawn_bonus[index] = max(0, value);
    }
                
    #[inline]
    pub fn get_passed_pawn_bonus(&self, index: usize) -> i32 {
        self.passed_pawn_bonus[index]
    }
    
    fn set_passed_pawn_king_defense_bonus(&mut self, index: usize, value: i32) {
        self.passed_pawn_king_defense_bonus[index] = max(0, value);
    }
                
    #[inline]
    pub fn get_passed_pawn_king_defense_bonus(&self, index: usize) -> i32 {
        self.passed_pawn_king_defense_bonus[index]
    }
    
    fn set_passed_pawn_king_attacked_penalty(&mut self, index: usize, value: i32) {
        self.passed_pawn_king_attacked_penalty[index] = max(0, value);
    }
                
    #[inline]
    pub fn get_passed_pawn_king_attacked_penalty(&self, index: usize) -> i32 {
        self.passed_pawn_king_attacked_penalty[index]
    }
    
    fn set_king_danger_piece_penalty(&mut self, index: usize, value: i32) {
        self.king_danger_piece_penalty[index] = value;
    }
        
    #[inline]
    pub fn get_king_danger_piece_penalty(&self, index: usize) -> i32 {
        self.king_danger_piece_penalty[index]
    }
    
    fn set_king_threat_adjustment(&mut self, index: usize, value: i32) {
        self.king_threat_adjustment[index] = value;
    }
        
    #[inline]
    pub fn get_king_threat_adjustment(&self, index: usize) -> i32 {
        self.king_threat_adjustment[index]
    }
    
    fn set_eg_knight_mob_bonus(&mut self, index: usize, value: i32) {
        self.eg_knight_mob_bonus[index] = value;
    }
        
    #[inline]
    pub fn get_eg_knight_mob_bonus(&self, index: usize) -> i32 {
        self.eg_knight_mob_bonus[index]
    }
    
    fn set_eg_bishop_mob_bonus(&mut self, index: usize, value: i32) {
        self.eg_bishop_mob_bonus[index] = value;
    }
        
    #[inline]
    pub fn get_eg_bishop_mob_bonus(&self, index: usize) -> i32 {
        self.eg_bishop_mob_bonus[index]
    }
    
    fn set_eg_rook_mob_bonus(&mut self, index: usize, value: i32) {
        self.eg_rook_mob_bonus[index] = value;
    }
        
    #[inline]
    pub fn get_eg_rook_mob_bonus(&self, index: usize) -> i32 {
        self.eg_rook_mob_bonus[index]
    }
    
    fn set_eg_queen_mob_bonus(&mut self, index: usize, value: i32) {
        self.eg_queen_mob_bonus[index] = value;
    }
        
    #[inline]
    pub fn get_eg_queen_mob_bonus(&self, index: usize) -> i32 {
        self.eg_queen_mob_bonus[index]
    }
    
    fn set_knight_mob_bonus(&mut self, index: usize, value: i32) {
        self.knight_mob_bonus[index] = value;
    }
        
    #[inline]
    pub fn get_knight_mob_bonus(&self, index: usize) -> i32 {
        self.knight_mob_bonus[index]
    }
    
    fn set_bishop_mob_bonus(&mut self, index: usize, value: i32) {
        self.bishop_mob_bonus[index] = value;
    }
        
    #[inline]
    pub fn get_bishop_mob_bonus(&self, index: usize) -> i32 {
        self.bishop_mob_bonus[index]
    }
    
    fn set_rook_mob_bonus(&mut self, index: usize, value: i32) {
        self.rook_mob_bonus[index] = value;
    }
        
    #[inline]
    pub fn get_rook_mob_bonus(&self, index: usize) -> i32 {
        self.rook_mob_bonus[index]
    }
    
    fn set_queen_mob_bonus(&mut self, index: usize, value: i32) {
        self.queen_mob_bonus[index] = value;
    }
        
    #[inline]
    pub fn get_queen_mob_bonus(&self, index: usize) -> i32 {
        self.queen_mob_bonus[index]
    }
    
    fn set_eg_pawn_pst(&mut self, index: usize, value: i32) {
        self.eg_pawn_pst[index] = value;
    }
        
    #[inline]
    pub fn get_eg_pawn_pst(&self) -> [i32; 64] {
        self.eg_pawn_pst
    }
            
    fn set_pawn_pst(&mut self, index: usize, value: i32) {
        self.pawn_pst[index] = value;
    }
        
    #[inline]
    pub fn get_pawn_pst(&self) -> [i32; 64] {
        self.pawn_pst
    }
            
    fn set_eg_knight_pst(&mut self, index: usize, value: i32) {
        self.eg_knight_pst[index] = value;
    }
        
    #[inline]
    pub fn get_eg_knight_pst(&self) -> [i32; 64] {
        self.eg_knight_pst
    }
            
    fn set_knight_pst(&mut self, index: usize, value: i32) {
        self.knight_pst[index] = value;
    }
        
    #[inline]
    pub fn get_knight_pst(&self) -> [i32; 64] {
        self.knight_pst
    }
            
    fn set_eg_bishop_pst(&mut self, index: usize, value: i32) {
        self.eg_bishop_pst[index] = value;
    }
        
    #[inline]
    pub fn get_eg_bishop_pst(&self) -> [i32; 64] {
        self.eg_bishop_pst
    }
            
    fn set_bishop_pst(&mut self, index: usize, value: i32) {
        self.bishop_pst[index] = value;
    }
        
    #[inline]
    pub fn get_bishop_pst(&self) -> [i32; 64] {
        self.bishop_pst
    }
            
    fn set_eg_rook_pst(&mut self, index: usize, value: i32) {
        self.eg_rook_pst[index] = value;
    }
        
    #[inline]
    pub fn get_eg_rook_pst(&self) -> [i32; 64] {
        self.eg_rook_pst
    }
            
    fn set_rook_pst(&mut self, index: usize, value: i32) {
        self.rook_pst[index] = value;
    }
        
    #[inline]
    pub fn get_rook_pst(&self) -> [i32; 64] {
        self.rook_pst
    }
            
    fn set_eg_queen_pst(&mut self, index: usize, value: i32) {
        self.eg_queen_pst[index] = value;
    }
        
    #[inline]
    pub fn get_eg_queen_pst(&self) -> [i32; 64] {
        self.eg_queen_pst
    }
            
    fn set_queen_pst(&mut self, index: usize, value: i32) {
        self.queen_pst[index] = value;
    }
        
    #[inline]
    pub fn get_queen_pst(&self) -> [i32; 64] {
        self.queen_pst
    }
            
    fn set_eg_king_pst(&mut self, index: usize, value: i32) {
        self.eg_king_pst[index] = value;
    }
        
    #[inline]
    pub fn get_eg_king_pst(&self) -> [i32; 64] {
        self.eg_king_pst
    }
            
    fn set_king_pst(&mut self, index: usize, value: i32) {
        self.king_pst[index] = value;
    }
        
    #[inline]
    pub fn get_king_pst(&self) -> [i32; 64] {
        self.king_pst
    }
            
}

const SINGLE_VALUE_OPTION_NAMES: [&'static str; 10] = ["castlingbonus", "lostqueensidecastlingpenalty", "lostkingsidecastlingpenalty", "doubledpawnpenalty", "kingshieldbonus", "pawncoverbonus", "knightkingthreat", "bishopkingthreat", "rookkingthreat", "queenkingthreat"];
const MULTI_VALUE_OPTION_NAMES: [&'static str; 26] = ["egpassedpawnbonus", "passedpawnbonus", "passedpawnkingdefensebonus", "passedpawnkingattackedpenalty", "kingdangerpiecepenalty", "kingthreatadjustment", "egknightmobbonus", "egbishopmobbonus", "egrookmobbonus", "egqueenmobbonus", "knightmobbonus", "bishopmobbonus", "rookmobbonus", "queenmobbonus", "egpawnpst", "pawnpst", "egknightpst", "knightpst", "egbishoppst", "bishoppst", "egrookpst", "rookpst", "egqueenpst", "queenpst", "egkingpst", "kingpst"];

pub fn parse_set_option(tx: &Sender<Message>, name: &str, value_str: &str) {
    if SINGLE_VALUE_OPTION_NAMES.contains(&name) {
        set_option_value(tx, name, value_str);
        return;
    }

    let name_without_index = name.replace(&['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'][..], "");
    if MULTI_VALUE_OPTION_NAMES.contains(&name_without_index.as_str()) {
        set_array_option_value(tx, name_without_index.as_str(), name, value_str);
        return;
    }
}

fn set_option_value(tx: &Sender<Message>, name: &str, value_str: &str) {
    let value = match i32::from_str(value_str) {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid int value: {}", value_str);
            return;
        }
    };

    send_message(tx, Message::SetOption(String::from(name), value));
}

fn set_array_option_value(tx: &Sender<Message>, name: &str, name_with_index: &str, value_str: &str) {
    let index = match i32::from_str(&name_with_index[name.len()..]) {
        Ok(index) => index,
        Err(_) => {
            println!("Invalid index: {}", name_with_index);
            return;
        }
    };

    let value = match i32::from_str(value_str) {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid int value: {}", value_str);
            return;
        }
    };

    send_message(tx, Message::SetArrayOption(String::from(name), index, value));
}

fn send_message(tx: &Sender<Message>, msg: Message) {
    match tx.send(msg) {
        Ok(_) => return,
        Err(err) => {
            eprintln!("could not send message to engine thread: {}", err);
        }
    }
}

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

