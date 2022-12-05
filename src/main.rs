use rurel::mdp::State;
use rurel::mdp::Agent;
use rurel::AgentTrainer;
use rurel::strategy::learn::QLearning;
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::terminate::TerminationStrategy;
use rand;
use std::time::Duration;
use std::time::Instant;

/*enum location {
    Entrance, Slingshot, Smashing, Wheel
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct StardewFairState { g: u64, frames: u64, tokens:u64}*/

#[derive(PartialEq, Eq, Hash, Clone)]
struct WeightedCoinState{balance:u8}

#[derive(PartialEq, Eq, Hash, Clone)]
struct WeightedCoinAction {bet:u8}

impl State for WeightedCoinState{
    type A = WeightedCoinAction;
    fn reward(&self) -> f64{
        if self.balance==100 {1.0}
        else {0.0}
    }
    fn actions(&self) -> Vec<WeightedCoinAction>{
        let bet_range = {
            if self.balance<50 {1..self.balance+1}
            else {1..(100-self.balance)+1}
        };
        return bet_range.map(|bet|WeightedCoinAction{bet:bet}).collect();
    }
}

struct WeightedCoinAgent{state: WeightedCoinState, weight:u8}
impl Agent<WeightedCoinState> for WeightedCoinAgent{
    fn current_state(&self) -> &WeightedCoinState {
		&self.state
	}
	fn take_action(&mut self, action: &WeightedCoinAction) -> () {
        self.state = WeightedCoinState { balance : 
            if rand::random::<u8>() <= self.weight {self.state.balance+action.bet}
            else {self.state.balance-action.bet}
        }
	}
}

struct WeightedCoinTermination{}
impl TerminationStrategy<WeightedCoinState> for WeightedCoinTermination{
    fn should_stop(&mut self, state:&WeightedCoinState) -> bool {
        if state.balance==0 || state.balance==100{
            return true;
        }
        if state.actions().len()==0{
            println!("{}",state.balance);
            return true;
        }
        return false;
    }
}

fn main() {
    let mut trainer=AgentTrainer::new();
    const TRIALS:usize=100000;
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
        if now.elapsed() > Duration::from_secs(21){
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
    }
}
