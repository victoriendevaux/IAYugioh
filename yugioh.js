const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');
let app = express(); // création de l'objet représentant notre application express
let port = process.env.PORT || 8080;

// Create the model
// Input
const input = tf.input({batchShape: [1, 10]});
// Hidden layer
const layer = tf.layers.dense({useBias: true, units: 32, activation: 'relu'}).apply(input);
// Output layer
const output = tf.layers.dense({useBias: true, units: 3, activation: 'linear'}).apply(layer);
// Create the model
const model = tf.model({inputs: input, outputs: output});
// Optimize
let model_optimizer = tf.train.adam(0.01);

app.use(bodyParser.json());

app.post('/', async (req, res) => {
    console.log("body    "+req.tabmonstres);
    console.log("body    "+req.body);
    console.log("body content    "+req.body.tabmonstres);
    const monstres = req.body.tabmonstres; // récupération des variables du body
    result = await pickAction(monstres, 1.1);
    console.log("resultat    "+result);
    console.log("resultat    "+result["monstreAttaque"]);
    console.log("resultat    "+result["monstreAttaquant"]);
    console.log("resultat    "+result["action"]);
    res.send(result);
});

// Pick an action eps-greedy
async function pickAction(st, eps)
{
    let st_tensor = tf.tensor([st]);
    let act;
    if (Math.random() > eps){ // Pick a random action
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
        return {"monstreAttaque": monstreAttaque, "monstreAttaquant": monstreAttaquant, "action": act};
    }
    else {
        monstreAttaque = null;
        monstreAttaquant=st[0];
        action = "RIEN";
        monstersPlayer=[]
        monstersEnnemies=[]


        for(i=0; i<5;i++) {

            if (st[0].monstre == null) {
                monstersPlayer[i] = 99999;
            } else {
                if (st[0].position === "attaque") {
                    monstersPlayer[i] = st[0].atk;
                } else {
                    monstersPlayer[i] = st[0].def;
                }
            }
        }

        for(i=0; i<5;i++) {
            valueIndex = i+ 5;
            if (st[valueIndex].monstre == null) {
                monstersEnnemies[i]=99999;

            } else {
                if (st[valueIndex].position === "attaque") {
                    monstersEnnemies[i]=st[valueIndex].atk;
                } else {
                    monstersEnnemies[i]=st[valueIndex].def;
                }
            }
        }

        console.log("monster in attaque mode player  "+  monstersPlayer);
        console.log("monster in defense mode for monster attack  "+  monstersEnnemies);

        x = tf.tensor1d(monstersEnnemies, 'int32');
        y = tf.tensor1d(monstersPlayer, 'int32');
        result = tf.less(x, y);

        console.log("compare value  "+  result);

        resultBoolean = Promise.resolve(result.data());
        console.log("compare value result boolean  "+  resultBoolean);

        return await resultBoolean.then(function(val) {
            console.log(val);
            for (index=0; index<5; index++) {
                if (val[index] === 1) {
                    monstreAttaque = st[5+index];
                    action = "ATTAQUER";
                }
            }

            console.log(monstreAttaquant);
            console.log(monstreAttaque);
            console.log(action);
            return {"monstreAttaque": monstreAttaque, "monstreAttaquant": monstreAttaquant, "action": action};
        });


        /*
        let result = model.predict(st_tensor);
        let argmax = result.argMax(1);

        console.log(result);
        act = argmax.buffer().values[0];
        argmax.dispose();
        result.dispose();

        st_tensor.dispose();
        return act; */
    }
}

function promiseResolveSystemAttack(atk, def, act){

}


// Return the mean of an array
function mean(array){
    if (array.length == 0)
        return null;
    return pickAction()
    return avg;
}


function train_model(states, actions, rewards, next_states){
    var size = next_states.length;

    // Transform each array into a tensor
    let tf_states = tf.tensor2d(states, shape=[states.length, 26]);
    let tf_rewards = tf.tensor2d(rewards, shape=[rewards.length, 1]);
    let tf_next_states = tf.tensor2d(next_states, shape=[next_states.length, 26]);
    let tf_actions = tf.tensor2d(actions, shape=[actions.length, 3]);
    // Get the list of loss to compute the mean later in this function
    let losses = []

    // Get the QTargets
    const Qtargets = tf.tidy(() => {
        let Q_stp1 = model.predict(tf_next_states);
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
