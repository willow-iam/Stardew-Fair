
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
use std::hash::Hash;
use std::time::Duration;
use std::time::Instant;
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
        if self.tokens==2000 {1.0}
        else {0.0}
    }
    fn actions(&self) -> Vec<StardewFairAction>{
        let mut actions = Vec::<StardewFairAction>::new();
        if self.current_location == location::Entrance {
            println!("Set of actions for going to other location");
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Slingshot});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Smashing});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Wheel});
        } else if self.current_location == location::Slingshot {
            println!("Set of actions for playing slingshot");
            println!("Also include a set of actions for going to other location");
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Entrance});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Smashing});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Wheel});

            actions.push(StardewFairAction { wheel_bet:0, play: true, next_location: location::Slingshot});
            //let result_range = 180 .. 240;    // # Tokens obtained from samples of speedrunners was to get
        } else if self.current_location == location::Smashing {
            println!("Set of actions for playing smashing");
            println!("Also include a set of actions for going to other location");
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Entrance});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Slingshot});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Wheel});

            actions.push(StardewFairAction { wheel_bet:0, play: true, next_location: location::Smashing});
        } else {
            println!("Set of actions for all possible bets for wheel");
            println!("Also include a set of actions for going to other location");
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Entrance});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Smashing});
            actions.push(StardewFairAction{wheel_bet:0,play:false,next_location:location::Slingshot});

            for bet in 1..self.tokens{
                actions.push(StardewFairAction{wheel_bet:bet,play:true,next_location:location::Wheel});
            }

        }
        return actions;
    }
}

fn main() {
    const TRIALS:usize=1000;
    /*let mut trainer=AgentTrainer::new();
    let now = Instant::now();
    let mut trial: u64=0;
    loop{
        let mut agent = WeightedCoinAgent {state: WeightedCoinState{balance:((1+trial%98) as u8)}, weight:55};
        trainer.train(
            &mut agent,
            &QLearning::new(0.2,0.0,2.0),
            &mut WeightedCoinTermination{},
            &RandomExploration::new()
        );
        trial+=1;
        if now.elapsed() > Duration::from_secs(20){
            println!("Trial {trial} complete");
            break;
        }
    }

    println!("Balance\tBet\tQ-value");
    for balance in 1..99{
        let state = WeightedCoinState{balance:balance};
        let action = trainer.best_action(&state).unwrap();
        println!("{}\t{}\t{}",
            balance,
            action.bet,
            trainer.expected_value(&state,&action).unwrap()
        );
    }*/

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


    let mut values = HashMap::new();
    for state in &state_space{
        //TODO:initial reward values
        values.insert(state.clone(),{if state.tokens>=2000 {1.0} else {0.0}});
    };

    let mut StardewFairValueIterator:ValueIterator<StardewFairState> = ValueIterator{
        state_space:state_space.clone(),
        values:values
    };

    let action_results = |state:&StardewFairState, action:&StardewFairAction|
    vec![];//TODO: transition model

    for trial in 1..TRIALS+1{
        StardewFairValueIterator.iterate(
            action_results
        );
        if trial % 100 == 0{
            println!("Trial {trial} complete");
        }
    }
    println!("Balance\tAction\tValue");
    for state in state_space{
        //TODO: Print actions for statespace
        //println!("{i}\t{}\t{}",StardewFairValueIterator.best_action(&state, action_results).bet,StardewFairValueIterator.value(state));
    }
    
}
