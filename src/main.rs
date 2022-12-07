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


#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct WeightedCoinState{balance:u8}

#[derive(PartialEq, Eq, Hash, Clone)]
struct WeightedCoinAction {bet:u8}

impl State for WeightedCoinState{
    type A = WeightedCoinAction;
    fn reward(&self) -> f64{
        if self.balance==100 {1.0}
        else {0.0}
    }
    //Set of available actions
    fn actions(&self) -> Vec<WeightedCoinAction>{
        let bet_range = {
            if self.balance<50 {1..self.balance+1}//Python: range(1,self.balance+1)
            else {1..(100-self.balance)+1}
        };
        return bet_range.map(|bet|WeightedCoinAction{bet:bet}).collect();
    }
}

struct WeightedCoinAgent{state: WeightedCoinState, weight:u8}
impl Agent<WeightedCoinState> for WeightedCoinAgent{
    //getter
    fn current_state(&self) -> &WeightedCoinState {
		&self.state
	}

	fn take_action(&mut self, action: &WeightedCoinAction) -> () {
        //Update the state to:
        self.state = WeightedCoinState { balance : 
            if rand::random::<u8>() <= self.weight {self.state.balance+action.bet}
            //If the coin is heads, balance + bet
            else {self.state.balance-action.bet}
            //If the coin is tails, balance - bet
        }
	}
}

/**
 * 
 */
struct WeightedCoinTermination{}
impl TerminationStrategy<WeightedCoinState> for WeightedCoinTermination{
    fn should_stop(&mut self, state:&WeightedCoinState) -> bool {
        if state.balance==0 || state.balance==100{
            return true;
        }

        //Error checking for other apparent sink states
        if state.actions().len()==0{
            println!("{}",state.balance);
            return true;
        }

        return false;
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

    let state_space:Vec<WeightedCoinState> = (0..101).map(|i|WeightedCoinState{balance:i}).collect();
    let mut map = HashMap::new();
    for state in &state_space.clone(){
        map.insert(*state,{if state.balance==100 {1.0} else {0.0}});
    };

    let mut CoinValueIterator:ValueIterator<WeightedCoinState> = ValueIterator{
        state_space:state_space,
        values:map
    };

    let action_results = |state:&WeightedCoinState, action:&WeightedCoinAction|
    vec![(weight,WeightedCoinState{balance:state.balance+action.bet}),
    (1.0-weight,WeightedCoinState{balance:state.balance-action.bet})];

    const weight:f64=0.25;
    for trial in 1..TRIALS+1{
        CoinValueIterator.iterate(
            action_results
        );
        if trial % 100 == 0{
            println!("Trial {trial} complete");
        }
    }
    println!("Balance\tAction\tValue");
    for i in 1..100{
        let state = WeightedCoinState{balance:i};
        println!("{i}\t{}\t{}",CoinValueIterator.best_action(&state, action_results).bet,CoinValueIterator.value(state));
    }
    
}
