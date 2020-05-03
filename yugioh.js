const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');
let app = express(); // création de l'objet représentant notre application express
let port = 8080;


// Create the model
// Input
const input = tf.input({batchShape: [1, 10]});
// Hidden layer
const layer = tf.layers.dense({useBias: true, units: 32, activation: 'relu'}).apply(input);
// Output layer
const output = tf.layers.dense({useBias: true, units: 2, activation: 'linear'}).apply(layer);
// Create the model
const model = tf.model({inputs: input, outputs: output});
// Optimize
let model_optimizer = tf.train.adam(0.01);

function model_loss(tf_states, tf_actions, Qtargets){
     return tf.tidy(() => {
         // valeur
         return model.predict(tf_states).sub(Qtargets).square().mul(tf_actions).mean();
     });
 }

app.use(bodyParser.json());

app.post('/', (req, res) => {
    const monstres = req.body; // récupération des variables du body
    random = play(monstres);
    res.send(train(monstres));
});

function play(monstre)
{
    return pickAction(monstre, 0.0);
}

function train(monstre) {
  let eps = 1.0;
  // Used to store the experiences
  let states = [];
  let rewards = [];
  let reward_mean = [];
  let next_states = [];
  let actions = [];


  for (let epi=0; epi < 150; epi++){
      let reward = 0;
      let step = 0;
      while (step < 400){
          // pick an action
          let act = pickAction(monstre, eps);
          reward = step +1
          st2 = step


          let mask = [0, 0];
          mask[act] = 1;

          // Randomly insert the new transition tuple
          let index = Math.floor(Math.random() * states.length);
          states.splice(index, 0, monstre);
          rewards.splice(index, 0, [reward]);
          reward_mean.splice(index, 0, reward)
          next_states.splice(index, 0, st2);
          actions.splice(index, 0, mask);
          // Be sure to keep the size of the dataset under 10000 transitions
          if (states.length > 10000){
              states = states.slice(1, states.length);
              rewards = rewards.slice(1, rewards.length);
              reward_mean = reward_mean.slice(1, reward_mean.length);
              next_states = next_states.slice(1, next_states.length);
              actions = actions.slice(1, actions.length);
          }

          //st = st2;
          step += 1;
      }
      // Decrease epsilon
      eps = Math.max(0.1, eps*0.99);

      // Train model every 5 episodes
      if (epi % 5 == 0){
          //console.log("---------------");
          //console.log("rewards mean", mean(reward_mean));
          //console.log("episode", epi);
           train_model(states, actions, rewards, next_states);
           tf.nextFrame();
      }
  }
}
// Pick an action eps-greedy
function pickAction(st, eps)
{
    let st_tensor = tf.tensor([st]);
    let act;
    let monstreAttaque
    let monstreAttaquant
    if (Math.random() < eps){ // Pick a random action
        act = "ATTAQUER";
        if (Math.floor(Math.random()*(100-0+1)+0) < 50) {
            act = "RIEN"
        }
        indexAttaque= Math.floor(Math.random()*(10-5+1)+5);
        monstreAttaque = st[indexAttaque];

        if (monstreAttaque != undefined && monstreAttaque.hasOwnProperty("monstre") && monstreAttaque.monstre != undefined && monstreAttaque.monstre == null) {
            act = "RIEN";
        }
        monstreAttaquant = st[0];
        //return {"monstreAttaque": monstreAttaque, "monstreAttaquant": monstreAttaquant, "action": act};
    }
    else {
        let result = model.predict(st_tensor);
        let argmax = result.argMax();
        return argmax;
        monstreAttaque = argmax.buffer().values(0);
        monstreAttaquant = st[0]
        act = "ATTAQUER";
        console.log(result);
        monstreAttaque.dispose();
        monstreAttaquant.dispose();
    }
    st_tensor.dispose();

    return {"monstreAttaque": monstreAttaque, "monstreAttaquant": monstreAttaquant, "action": act};
}


// Return the mean of an array
/*function mean(array){
    if (array.length == 0)
        return null;
    return pickAction()
    return avg;
}*/


function train_model(states, actions, rewards, next_states){
    var size = next_states.length;

    // Transform each array into a tensor
    let tf_states = tf.tensor2d(states, shape=[states.length, 10]);
    let tf_rewards = tf.tensor2d(rewards, shape=[rewards.length, 1]);
    let tf_next_states = tf.tensor2d(next_states, shape=[next_states.length, 1]);
    let tf_actions = tf.tensor2d(actions, shape=[actions.length, 2]);
    // Get the list of loss to compute the mean later in this function
    let losses = []

    // Get the QTargets
    const Qtargets = tf.tidy(() => {
        let Q_stp1 = model.predict(tf_states);
        let Qtargets = tf.tensor2d(Q_stp1.max(1).expandDims(1).mul(tf.scalar(0.99)).add(tf_rewards).buffer().values, shape=[size, 1]);
        return Qtargets;
    });

    // Generate batch of training and train the model
    let batch_size = 32;
    for (var b = 0; b < size; b+=32) {

        // Select the batch
        let to = (b + batch_size < size) ?  batch_size  : (size - b);
        const tf_states_b = tf_states.slice(b, to);
        const tf_actions_b = tf_actions.slice(b, to);
        const Qtargets_b = Qtargets.slice(b, to);

        // Minimize the error
        model_optimizer.minimize(() => {
            const loss = model_loss(tf_states_b, tf_actions_b, Qtargets_b);
            losses.push(loss.buffer().values[0]);
            return loss;
        });

        // Dispose the tensors from the memory
        tf_states_b.dispose();
        tf_actions_b.dispose();
        Qtargets_b.dispose();
    }

    console.log("Mean loss", mean(losses));

    // Dispose the tensors from the memory
    Qtargets.dispose();
    tf_states.dispose();
    tf_rewards.dispose();
    tf_next_states.dispose();
    tf_actions.dispose();
}


app.listen(port, () =>  { // ecoute du serveur sur le port 8080
    console.log('le serveur fonctionne')
})
