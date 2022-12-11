#![allow(warnings)]
use rurel::mdp::State;
use rurel::mdp::Agent;
use rurel::AgentTrainer;
use rurel::strategy::learn::QLearning;
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::terminate::TerminationStrategy;
use rand;
use stardew_valley_fair::value_iteration::StatespaceIterator;
use stardew_valley_fair::value_iteration::ValueIterator;
use std::collections::HashMap;
use std::fmt::Display;
use std::fmt::write;
use std::hash::Hash;
use std::panic::Location;
use std::thread::LocalKey;
use std::time::Duration;
use std::time::Instant;
use std::fs::{File, OpenOptions};
use std::io::{Write, BufReader, BufRead, Error};
mod value_iteration;

#[derive(PartialEq, Eq, Hash, Clone)]
enum location {
    Entrance, Slingshot, Smashing, Wheel
}


#[derive(PartialEq, Eq, Hash, Clone)]
struct StardewFairState { g: u64, seconds: u64, tokens: u64, current_location :location }


/// All fields must be 0 except 1. Not a union because rust doesn't let you derive
/// stuff from unions.
#[derive(PartialEq, Eq, Hash, Clone)]
struct StardewFairAction {wheel_bet :u64, play :bool, next_location :location}

impl State for StardewFairState{
    type A = StardewFairAction;
    fn reward(&self) -> f64{
        if (self.tokens>=2000 && self.current_location==location::Entrance && self.seconds<266) {1.0}
        else {0.0}
    }
    fn actions(&self) -> Vec<StardewFairAction>{
        if(self.tokens>=2000 && self.current_location==location::Entrance) || self.seconds>=266 {
            return Vec::new();
        }
        let mut actions = Vec::<StardewFairAction>::new();
        if self.current_location == location::Entrance {
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Slingshot});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Smashing});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Wheel});
        } else if self.current_location == location::Slingshot {
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Entrance});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Smashing});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Wheel});
            if(self.g>=50){
                actions.push(StardewFairAction { wheel_bet:0, play: true, next_location: location::Slingshot});
            }
            //let result_range = 180 .. 240;    // # Tokens obtained from samples of speedrunners was to get
        } else if self.current_location == location::Smashing {
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Entrance});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Slingshot});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Wheel});

            actions.push(StardewFairAction { wheel_bet:0, play: true, next_location: location::Smashing});
        } else {
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Entrance});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Smashing});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Slingshot});

            for bet in 1..self.tokens{
                if (self.tokens<222 && bet%11==0) || (self.tokens>222 && bet%111==0) {
                    actions.push(StardewFairAction{wheel_bet:bet,play:true,next_location:location::Wheel});
                }
            }

        }
        return actions;
    }
}

fn main() {
    println!("Begin setup");
    const TRIALS:usize=1000;
    let mut state_space:Vec<StardewFairState>=Vec::new();//TODO: State space
    let mut gold = 0;

    while gold < 500 {
        for secs in 0..330 {
            for t in 0..2241 {
                state_space.push(StardewFairState{g:gold, seconds:secs, tokens:t, current_location:location::Entrance});
                state_space.push(StardewFairState{g:gold, seconds:secs, tokens:t, current_location:location::Slingshot});
                state_space.push(StardewFairState{g:gold, seconds:secs, tokens:t, current_location:location::Smashing});
                state_space.push(StardewFairState{g:gold, seconds:secs, tokens:t, current_location:location::Wheel});
            }
        }
        gold = gold + 50;
    }
    println!("Defined statespace");

    let mut values = HashMap::new();
    for state in state_space.clone(){
        //TODO:initial reward values
        values.insert(state.clone(),{if state.tokens>=2000 {1.0} else {0.0}});
    };

    let mut StardewFairValueIterator:ValueIterator<StardewFairState> = ValueIterator{
        state_space:state_space.clone(),
        values:values
    };

    let action_results = |state:&StardewFairState, action:&StardewFairAction|{
        if state.current_location == location::Entrance { // ENTRANCE
            if action.next_location == location::Slingshot {
                vec![(1.0,StardewFairState{seconds:state.seconds + 5, g:state.g, 
                    tokens:state.tokens, current_location:location::Slingshot}),  // 5 seconds entrance to slingshot
                    ]
            }
            else if action.next_location == location::Smashing {
                vec![(1.0,StardewFairState{seconds:state.seconds + 3, g:state.g, 
                    tokens:state.tokens, current_location:location::Smashing}),  // 3 seconds entrance to smashing
                    ]       
            }
            else if action.next_location == location::Wheel {
                vec![(1.0,StardewFairState{seconds:state.seconds + 6, g:state.g, 
                    tokens:state.tokens, current_location:location::Wheel}),  // 6 seconds entrance to wheel
                    ]      
            }
            else{
                panic!("Invalid action")
            }
        }
        else if state.current_location == location::Slingshot{ // SLINGSHOT
            if action.next_location == location::Entrance {
                vec![(1.0,StardewFairState{seconds:state.seconds + 5, g:state.g, 
                    tokens:state.tokens, current_location:location::Entrance}),  // 5 seconds slingshot to entrance
                    ]   
            }
            else if action.next_location == location::Smashing {
                vec![(1.0,StardewFairState{seconds:state.seconds + 3, g:state.g, 
                    tokens:state.tokens, current_location:location::Smashing}),  // 3 seconds slingshot to smashing
                    ]  
            }
            else if action.next_location == location::Wheel {
                vec![(1.0,StardewFairState{seconds:state.seconds + 3, g:state.g, 
                    tokens:state.tokens, current_location:location::Wheel}),  // 3 seconds slingshot to wheel
                    ]         
            }
            else if action.play == true {
                
                vec![(0.5,StardewFairState{seconds:state.seconds + 63, g:state.g-50, 
                    tokens:state.tokens+500, current_location:location::Slingshot}),  // 63 seconds to play slingshot
                    (0.5,StardewFairState{seconds:state.seconds, g:state.g-50, 
                    tokens:state.tokens+200, current_location:location::Slingshot})] 
            }
            else{
                panic!("Invalid action")
            }
        }
        else if state.current_location == location::Smashing{  // SMASHING STONE
            if action.next_location == location::Entrance {
                vec![(1.0,StardewFairState{seconds:state.seconds + 3, g:state.g, 
                    tokens:state.tokens, current_location: location::Entrance}),  // 3 seconds smashing to entrance
                    ]   
            }
            else if action.next_location == location::Slingshot {
                vec![(1.0,StardewFairState{seconds:state.seconds + 3, g:state.g, 
                    tokens:state.tokens, current_location:location::Slingshot}),  // 3 seconds smashing to slingshot
                    ]   
            }
            else if action.next_location == location::Wheel {
                vec![(1.0,StardewFairState{seconds:state.seconds + 4, g:state.g, 
                    tokens:state.tokens, current_location:location::Wheel}),  // 4 seconds smashing to wheel
                    ]   
            }
            else if action.play == true {
                vec![(0.1,StardewFairState{seconds:state.seconds + 3, g:state.g, 
                    tokens:state.tokens, current_location:location::Smashing}),  // 3 seconds to play smashing
                    (0.9,StardewFairState{seconds:state.seconds+3, g:state.g, 
                    tokens:state.tokens+1, current_location:location::Smashing})
                    ]        
            }
            else{
                panic!("Invalid action")
            }
        }
        else { // WHEEL
            if action.next_location == location::Entrance {
                vec![(1.0,StardewFairState{seconds:state.seconds + 6, g:state.g, 
                    tokens:state.tokens, current_location: location::Entrance}),  // 6 seconds wheel to entrance
                    ]
            }
            else if action.next_location == location::Slingshot {
                vec![(1.0,StardewFairState{seconds:state.seconds + 3, g:state.g, 
                    tokens:state.tokens, current_location:location::Slingshot}),  // 3 seconds wheel to slingshot
                    ]
            }
            else if action.next_location == location::Smashing {
                vec![(1.0,StardewFairState{seconds:state.seconds + 4, g:state.g, 
                    tokens:state.tokens, current_location:location::Smashing}),  // 4 seconds wheel to smashing
                    ]
            }
            else if action.play == true {
                vec![(0.75,StardewFairState{seconds:state.seconds+16, g:state.g, 
                    tokens:state.tokens+action.wheel_bet, current_location:location::Wheel}),  // 16 seconds to play wheel
                    (0.25,StardewFairState{seconds:state.seconds, g:state.g, 
                    tokens:state.tokens-action.wheel_bet, current_location:location::Wheel})
                    ]
            }
            else{
                panic!("Invalid action")
            }
        }
    };

    println!("Finished setup");
    for trial in 0..TRIALS+1{
        println!("Writing to file");
        let file_name = format!("output_{trial}.txt");
        let mut output = File::create(file_name).unwrap();
        for state in &state_space{
            //TODO: Print actions for statespace
            //println!("{i}\t{}\t{}",StardewFairValueIterator.best_action(&state, action_results).bet,StardewFairValueIterator.value(state));
            let tokens =state.tokens;
            let g = state.g;
            let location=&state.current_location;
            let value = StardewFairValueIterator.value(state.clone());
            let mut wheel_bet=99;
            let mut play = false;
            let mut next_location = 
                match state.current_location{
                    (location::Entrance) => {"Entrance"}
                    (location::Slingshot) => {"Slingshot"}
                    (location::Smashing) => {"Smashing"}
                    (location::Wheel) => {"Wheel"}
                };
                if !state.actions().is_empty() {
                    let action = StardewFairValueIterator.best_action(state, action_results);
                    wheel_bet=action.wheel_bet;
                    play=action.play;
                    next_location = {
                        match action.next_location{
                            (location::Entrance) => {"Entrance"}
                            (location::Slingshot) => {"Slingshot"}
                            (location::Smashing) => {"Smashing"}
                            (location::Wheel) => {"Wheel"}
                        }
                    }
                };

            

            match location{
                (location::Entrance) => {
                    writeln!(output, "Tokens:{tokens}\tg:{g}\tLocation:Entrance\tvalue:{value}\t");
                    writeln!(output, "\tnext_location:{next_location}");

                }
                (location::Slingshot) => {
                    writeln!(output, "Tokens:{tokens}\tg:{g}\tLocation:Slingshot\tvalue:{value}\t");
                    if(play){writeln!(output,"\tPlay");}
                    else{writeln!(output, "\tnext_location:{next_location}");}

                }
                (location::Smashing) => {
                    writeln!(output, "Tokens:{tokens}\tg:{g}\tLocation:Smashing\tvalue:{value}\t");
                    if(play){writeln!(output,"\tPlay");}
                    else{writeln!(output, "\tnext_location:{next_location}");}
                }
                (location::Wheel) => {
                    writeln!(output, "Tokens:{tokens}\tg:{g}\tLocation:Wheel\tvalue:{value}\t");
                    if(play){writeln!(output,"\tBet {wheel_bet}");}
                    else{writeln!(output, "\tnext_location:{next_location}");}

                }
            }

        }
        StardewFairValueIterator.iterate(
            action_results
        );
        if trial % 1 == 0{
            println!("Trial {trial} complete");
        }
        

    }
    
}